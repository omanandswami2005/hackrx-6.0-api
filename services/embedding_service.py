# services/embedding_service.py - Text Embedding Service
"""
Text embedding and similarity computation service using sentence transformers.
Team member: ML/Embeddings Lead
"""

import asyncio
import time
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import EmbeddingServiceStats
from services.document_service import DocumentService
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """Service for generating embeddings and computing similarities"""
    
    def __init__(self):
        self.config = Config()
        self.model: Optional[SentenceTransformer] = None
        self.document_service = DocumentService()
        self.stats = EmbeddingServiceStats(
            requests_processed=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            error_count=0,
            chunks_processed=0,
            embeddings_generated=0,
            cache_hits=0,
            cache_misses=0
        )
        self._embedding_cache = {}  # Simple in-memory cache for embeddings
    
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_model_sync
            )
            
            logger.info("✅ Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    
    def _load_model_sync(self) -> SentenceTransformer:
        """Load model synchronously (runs in thread pool)"""
        model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        model.max_seq_length = self.config.EMBEDDING_MAX_SEQ_LENGTH
        return model
    
    async def warmup(self):
        """Warm up the model with a sample embedding"""
        if not self.model:
            await self.initialize()
        
        try:
            logger.info("Warming up embedding model...")
            
            # Create sample embedding to warm up GPU/CPU
            sample_texts = [
                "This is a sample text for warming up the embedding model.",
                "Another sample sentence to ensure proper initialization."
            ]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.model.encode, sample_texts
            )
            
            logger.info("✅ Embedding model warmed up")
            
        except Exception as e:
            logger.error(f"❌ Model warmup failed: {e}")
            raise
    
    async def create_document_embeddings(self, text_content: str) -> Tuple[List[str], np.ndarray]:
        """
        Create embeddings for document chunks.
        
        Args:
            text_content: Full document text
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        start_time = time.time()
        
        try:
            # Split text into chunks
            chunks = self.document_service.chunk_text(text_content)
            
            if not chunks:
                raise ValueError("No valid chunks created from document")
            
            logger.info(f"Created {len(chunks)} chunks for embedding")
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(chunks)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(chunks), success=True)
            
            logger.info(f"Generated embeddings in {processing_time:.2f}s")
            
            return chunks, embeddings
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, 0, success=False)
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        if cache_key in self._embedding_cache:
            self.stats.cache_hits += 1
            logger.debug("Using cached embeddings")
            return self._embedding_cache[cache_key]
        
        self.stats.cache_misses += 1
        
        try:
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._encode_texts_batch, texts
            )
            
            # Cache the results
            self._embedding_cache[cache_key] = embeddings
            self.stats.embeddings_generated += len(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def _encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches (runs in thread pool)"""
        # Process in batches to manage memory
        batch_size = self.config.BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    async def find_relevant_chunks(
        self, 
        question: str, 
        chunks: List[str], 
        chunk_embeddings: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Find most relevant chunks for a question.
        
        Args:
            question: Query question
            chunks: List of text chunks
            chunk_embeddings: Precomputed chunk embeddings
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        try:
            # Generate question embedding
            question_embedding = await self._generate_embeddings([question])
            
            # Compute similarities
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            
            # Get top-k most similar chunks
            top_k_indices = np.argsort(similarities)[-self.config.TOP_K_CHUNKS:][::-1]
            
            # Filter by minimum similarity threshold
            relevant_chunks = []
            for idx in top_k_indices:
                similarity = similarities[idx]
                if similarity >= self.config.MIN_SIMILARITY_THRESHOLD:
                    relevant_chunks.append((chunks[idx], float(similarity)))
            
            logger.debug(f"Found {len(relevant_chunks)} relevant chunks for question")
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Chunk retrieval error: {e}")
            raise RuntimeError(f"Failed to find relevant chunks: {str(e)}")
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts"""
        # Simple hash-based key - could be improved with better hashing
        import hashlib
        combined_text = "|||".join(texts)
        return hashlib.md5(combined_text.encode()).hexdigest()
    
    def _update_stats(self, processing_time: float, chunks_count: int, success: bool = True):
        """Update service statistics"""
        self.stats.requests_processed += 1
        self.stats.total_processing_time += processing_time
        self.stats.average_processing_time = (
            self.stats.total_processing_time / self.stats.requests_processed
        )
        self.stats.last_request_time = time.time()
        
        if success:
            self.stats.chunks_processed += chunks_count
        else:
            self.stats.error_count += 1
    
    def get_stats(self) -> dict:
        """Get service statistics"""
        stats_dict = self.stats.dict()
        stats_dict.update({
            "cache_size": len(self._embedding_cache),
            "model_loaded": self.model is not None,
            "cache_hit_rate": (
                self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses)
                if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0.0
            )
        })
        return stats_dict
    
    def clear_cache(self):
        """Clear embedding cache"""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"Cleared embedding cache ({cache_size} entries)")
    
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            if not self.model:
                return False
            
            # Quick test embedding
            test_embedding = await self._generate_embeddings(["health check"])
            return test_embedding is not None and len(test_embedding) > 0
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False