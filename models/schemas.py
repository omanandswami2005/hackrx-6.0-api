# models/schemas.py - Pydantic Models and Schemas
"""
Data models and validation schemas.
Team member: Backend/API Lead or Data Structures Lead
"""

from pydantic import BaseModel, HttpUrl, validator
from typing import List, Dict, Any, Optional
import time

class HackRxRequest(BaseModel):
    """Request model for the main API endpoint"""
    documents: str  # URL to PDF document
    questions: List[str]
    
    @validator('documents')
    def validate_document_url(cls, v):
        """Validate document URL format"""
        if not v or not v.strip():
            raise ValueError('Document URL cannot be empty')
        
        # Basic URL validation
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('Document URL must start with http:// or https://')
        
        return v.strip()
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list"""
        if not v:
            raise ValueError('Questions list cannot be empty')
        
        if len(v) > 50:
            raise ValueError('Maximum 50 questions allowed')
        
        # Filter out empty questions
        valid_questions = [q.strip() for q in v if q and q.strip()]
        
        if not valid_questions:
            raise ValueError('At least one valid question is required')
        
        return valid_questions
    
    class Config:
        json_schema_extra  = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the main topic of this document?",
                    "What are the key findings mentioned?"
                ]
            }
        }

class HackRxResponse(BaseModel):
    """Response model for the main API endpoint"""
    answers: List[str]
    
    class Config:
        json_schema_extra  = {
            "example": {
                "answers": [
                    "The main topic is artificial intelligence applications.",
                    "Key findings include improved efficiency and accuracy."
                ]
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    services: Dict[str, bool]
    timestamp: float
    
    class Config:
        json_schema_extra  = {
            "example": {
                "status": "healthy",
                "services": {
                    "document_service": True,
                    "embedding_service": True,
                    "ai_service": True,
                    "cache_service": True
                },
                "timestamp": 1642680000.0
            }
        }

class DocumentChunk(BaseModel):
    """Model for document text chunks"""
    text: str
    chunk_id: int
    start_position: int
    end_position: int
    similarity_score: Optional[float] = None
    
    def __len__(self) -> int:
        return len(self.text.split())

class EmbeddingResult(BaseModel):
    """Model for embedding computation results"""
    chunks: List[str]
    embeddings: List[List[float]]  # Will be converted to numpy array in service
    processing_time: float
    chunk_count: int

class AIResponse(BaseModel):
    """Model for AI service responses"""
    answer: str
    confidence: Optional[float] = None
    context_used: List[str]
    processing_time: float
    tokens_used: Optional[int] = None

class CacheEntry(BaseModel):
    """Model for cache entries"""
    key: str
    value: str
    created_at: float
    expires_at: float
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

class ServiceStats(BaseModel):
    """Base model for service statistics"""
    requests_processed: int
    total_processing_time: float
    average_processing_time: float
    error_count: int
    last_request_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        if self.requests_processed == 0:
            return 0.0
        return (self.requests_processed - self.error_count) / self.requests_processed

class DocumentServiceStats(ServiceStats):
    """Statistics for document service"""
    documents_processed: int
    total_pages_processed: int
    total_text_extracted: int  # Total characters
    pdf_errors: int

class EmbeddingServiceStats(ServiceStats):
    """Statistics for embedding service"""
    chunks_processed: int
    embeddings_generated: int
    cache_hits: int
    cache_misses: int

class AIServiceStats(ServiceStats):
    """Statistics for AI service"""
    questions_answered: int
    total_tokens_used: int
    average_response_length: float
    gemini_api_errors: int

class CacheServiceStats(ServiceStats):
    """Statistics for cache service"""
    cache_hits: int
    cache_misses: int
    cache_size: int
    memory_usage: Optional[int] = None  # In bytes
    
    @property
    def hit_rate(self) -> float:
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests

class ProcessingMetrics(BaseModel):
    """Overall processing metrics"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    document_url: str
    question_count: int
    cache_hits: int
    total_chunks: int
    
    @property
    def processing_time(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

# Error Models
class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: str
    timestamp: float = time.time()
    request_id: Optional[str] = None

class ValidationErrorResponse(BaseModel):
    """Validation error response model"""
    error: str = "Validation Error"
    details: List[Dict[str, Any]]
    timestamp: float = time.time()