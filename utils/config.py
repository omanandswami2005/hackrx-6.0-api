# utils/config.py - Configuration Management
"""
Centralized configuration management.
Team member: DevOps/Configuration Lead
"""

import os
from typing import Optional

class Config:
    """Application configuration settings"""
    
    # API Configuration
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", 50))
    
    # AI Model Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    AI_TEMPERATURE: float = float(os.getenv("AI_TEMPERATURE", 0.1))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", 200))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_MAX_SEQ_LENGTH: int = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", 256))
    
    # Document Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 300))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", 3))
    MIN_SIMILARITY_THRESHOLD: float = float(os.getenv("MIN_SIMILARITY_THRESHOLD", 0.1))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", 2000))
    
    # Request Configuration
    DOWNLOAD_TIMEOUT: int = int(os.getenv("DOWNLOAD_TIMEOUT", 15))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
    
    # Cache Configuration
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", 10))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        if cls.CHUNK_SIZE <= cls.CHUNK_OVERLAP:
            raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
        
        if cls.TOP_K_CHUNKS <= 0:
            raise ValueError("TOP_K_CHUNKS must be positive")
    
    @classmethod
    def get_environment_info(cls) -> dict:
        """Get current environment configuration info"""
        return {
            "port": cls.PORT,
            "debug": cls.DEBUG,
            "gemini_model": cls.GEMINI_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "top_k_chunks": cls.TOP_K_CHUNKS,
            "cache_enabled": cls.ENABLE_CACHE,
            "max_questions": cls.MAX_QUESTIONS,
        }

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    CACHE_TTL = 300  # 5 minutes for development

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    MAX_CONCURRENT_REQUESTS = 20
    CACHE_TTL = 7200  # 2 hours for production

class TestConfig(Config):
    """Test environment configuration"""
    DEBUG = True
    ENABLE_CACHE = False
    MAX_QUESTIONS = 10

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()