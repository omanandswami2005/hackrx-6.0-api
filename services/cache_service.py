# services/cache_service.py - Caching Service
"""
Redis-based caching service for improved performance.
Team member: DevOps/Caching Lead
"""

import asyncio
import json
import hashlib
import time
from typing import List, Optional, Dict, Any
import aioredis
from aioredis import Redis

from models.schemas import CacheServiceStats
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CacheService:
    """Service for caching answers and embeddings using Redis"""
    
    def __init__(self):
        self.config = Config()
        self.redis: Optional[Redis] = None
        self.fallback_cache: Dict[str, Any] = {}  # In-memory fallback
        self.stats = CacheServiceStats(
            requests_processed=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            error_count=0,
            cache_hits=0,
            cache_misses=0,
            cache_size=0
        )
        
        # Initialize Redis connection
        if self.config.ENABLE_CACHE:
            asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        if not self.config.ENABLE_CACHE:
            logger.info("Caching disabled")
            return
        
        try:
            # Try Redis URL first, then host/port
            if self.config.REDIS_URL:
                self.redis = await aioredis.from_url(
                    self.config.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0
                )
            else:
                self.redis = await aioredis.Redis(
                    host=self.config.REDIS_HOST,
                    port=self.config.REDIS_PORT,
                    db=self.config.REDIS_DB,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0
                )
            
            # Test connection
            await self.redis.ping()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed, using in-memory cache: {e}")
            self.redis = None
    
    def _generate_cache_key(self, document_url: str, question: str) -> str:
        """Generate a unique cache key for document-question pair"""
        # Create a deterministic key based on URL and question
        key_string = f"{document_url}:{question}"
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"hackrx:qa:{key_hash}"
    
    def _generate_document_key(self, document_url: str) -> str:
        """Generate cache key for document text"""
        url_hash = hashlib.sha256(document_url.encode()).hexdigest()[:16]
        return f"hackrx:doc:{url_hash}"
    
    async def get_cached_answer(self, document_url: str, question: str) -> Optional[str]:
        """Get cached answer for a specific question"""
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key(document_url, question)
            
            # Try Redis first
            if self.redis:
                try:
                    cached_data = await self.redis.get(cache_key)
                    if cached_data:
                        answer_data = json.loads(cached_data)
                        self.stats.cache_hits += 1
                        logger.debug(f"Cache hit for question: {question[:50]}...")
                        return answer_data.get("answer")
                except Exception as e:
                    logger.warning(f"Redis get error: {e}")
            
            # Fallback to in-memory cache
            if cache_key in self.fallback_cache:
                cache_entry = self.fallback_cache[cache_key]
                if cache_entry["expires_at"] > time.time():
                    self.stats.cache_hits += 1
                    logger.debug(f"Memory cache hit for question: {question[:50]}...")
                    return cache_entry["answer"]
                else:
                    # Remove expired entry
                    del self.fallback_cache[cache_key]
            
            self.stats.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats.error_count += 1
            return None
        finally:
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
    
    async def get_cached_answers(self, document_url: str, questions: List[str]) -> List[Optional[str]]:
        """Get cached answers for multiple questions"""
        tasks = [
            self.get_cached_answer(document_url, question)
            for question in questions
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def cache_answer(self, document_url: str, question: str, answer: str):
        """Cache an answer for a specific question"""
        try:
            cache_key = self._generate_cache_key(document_url, question)
            
            cache_data = {
                "answer": answer,
                "cached_at": time.time(),
                "document_url": document_url,
                "question": question
            }
            
            # Try Redis first
            if self.redis:
                try:
                    await self.redis.setex(
                        cache_key,
                        self.config.CACHE_TTL,
                        json.dumps(cache_data)
                    )
                    logger.debug(f"Cached answer in Redis: {question[:50]}...")
                    return
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
            
            # Fallback to in-memory cache
            self.fallback_cache[cache_key] = {
                "answer": answer,
                "expires_at": time.time() + self.config.CACHE_TTL
            }
            
            # Limit in-memory cache size
            if len(self.fallback_cache) > 1000:
                self._cleanup_memory_cache()
            
            logger.debug(f"Cached answer in memory: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats.error_count += 1
    
    async def cache_answers(self, document_url: str, questions: List[str], answers: List[str]):
        """Cache multiple answers"""
        tasks = [
            self.cache_answer(document_url, question, answer)
            for question, answer in zip(questions, answers)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def cache_document_text(self, document_url: str, text_content: str):
        """Cache extracted document text"""
        try:
            cache_key = self._generate_document_key(document_url)
            
            cache_data = {
                "text": text_content,
                "cached_at": time.time(),
                "url": document_url
            }
            
            # Only cache in Redis due to size
            if self.redis:
                try:
                    await self.redis.setex(
                        cache_key,
                        self.config.CACHE_TTL * 2,  # Cache documents longer
                        json.dumps(cache_data)
                    )
                    logger.debug(f"Cached document text: {len(text_content)} chars")
                except Exception as e:
                    logger.warning(f"Redis document cache error: {e}")
                    
        except Exception as e:
            logger.error(f"Document cache error: {e}")
            self.stats.error_count += 1
    
    async def get_cached_document_text(self, document_url: str) -> Optional[str]:
        """Get cached document text"""
        try:
            cache_key = self._generate_document_key(document_url)
            
            if self.redis:
                try:
                    cached_data = await self.redis.get(cache_key)
                    if cached_data:
                        doc_data = json.loads(cached_data)
                        logger.debug("Document text cache hit")
                        return doc_data.get("text")
                except Exception as e:
                    logger.warning(f"Redis document get error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Document cache get error: {e}")
            return None
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.fallback_cache.items()
            if entry["expires_at"] <= current_time
        ]
        
        for key in expired_keys:
            del self.fallback_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.fallback_cache) > 500:
            sorted_items = sorted(
                self.fallback_cache.items(),
                key=lambda x: x[1]["expires_at"]
            )
            
            # Keep only the newest 500 entries
            self.fallback_cache = dict(sorted_items[-500:])
        
        logger.debug(f"Cleaned memory cache, {len(expired_keys)} expired entries removed")
    
    async def invalidate_document_cache(self, document_url: str):
        """Invalidate all cache entries for a specific document"""
        try:
            # Get document-specific pattern
            url_hash = hashlib.sha256(document_url.encode()).hexdigest()[:16]
            
            if self.redis:
                try:
                    # Find and delete all keys matching the pattern
                    pattern = f"hackrx:*:{url_hash}*"
                    keys = await self.redis.keys(pattern)
                    
                    if keys:
                        await self.redis.delete(*keys)
                        logger.info(f"Invalidated {len(keys)} cache entries for document")
                except Exception as e:
                    logger.warning(f"Redis invalidation error: {e}")
            
            # Clean memory cache
            keys_to_remove = [
                key for key in self.fallback_cache.keys()
                if url_hash in key
            ]
            
            for key in keys_to_remove:
                del self.fallback_cache[key]
            
            logger.info(f"Invalidated cache for document: {document_url}")
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            self.stats.error_count += 1
    
    async def clear_all_cache(self):
        """Clear all cache entries"""
        try:
            if self.redis:
                try:
                    keys = await self.redis.keys("hackrx:*")
                    if keys:
                        await self.redis.delete(*keys)
                        logger.info(f"Cleared {len(keys)} Redis cache entries")
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")
            
            # Clear memory cache
            cache_size = len(self.fallback_cache)
            self.fallback_cache.clear()
            logger.info(f"Cleared {cache_size} memory cache entries")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats.error_count += 1
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        info = {
            "redis_connected": self.redis is not None,
            "memory_cache_size": len(self.fallback_cache),
            "cache_enabled": self.config.ENABLE_CACHE,
        }
        
        if self.redis:
            try:
                redis_info = await self.redis.info("memory")
                hackrx_keys = await self.redis.keys("hackrx:*")
                
                info.update({
                    "redis_memory_usage": redis_info.get("used_memory", 0),
                    "redis_hackrx_keys": len(hackrx_keys),
                    "redis_connected": True
                })
            except Exception as e:
                logger.warning(f"Redis info error: {e}")
                info["redis_connected"] = False
        
        return info
    
    def _update_stats(self, processing_time: float):
        """Update service statistics"""
        self.stats.requests_processed += 1
        self.stats.total_processing_time += processing_time
        self.stats.average_processing_time = (
            self.stats.total_processing_time / self.stats.requests_processed
        )
        self.stats.last_request_time = time.time()
        self.stats.cache_size = len(self.fallback_cache)
    
    async def get_stats(self) -> dict:
        """Get service statistics"""
        stats_dict = self.stats.dict()
        
        # Add cache info
        cache_info = await self.get_cache_info()
        stats_dict.update(cache_info)
        
        # Calculate hit rate
        total_requests = self.stats.cache_hits + self.stats.cache_misses
        hit_rate = self.stats.cache_hits / total_requests if total_requests > 0 else 0.0
        stats_dict["hit_rate"] = hit_rate
        
        return stats_dict
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            if self.redis:
                await self.redis.close()
                logger.info("Redis connection closed")
            
            self.fallback_cache.clear()
            logger.info("Cache service cleaned up")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    async def health_check(self) -> bool:
        """Check if cache service is healthy"""
        try:
            if self.redis:
                await self.redis.ping()
                return True
            else:
                # If Redis is not available, in-memory cache is still functional
                return True
                
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False