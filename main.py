# main.py - Main FastAPI Application
"""
Main application entry point.
Team member: Backend/API Lead
"""

import os
import time
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from contextlib import asynccontextmanager

# Import modular components
from models.schemas import HackRxRequest, HackRxResponse, HealthResponse
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.ai_service import AIService
from services.cache_service import CacheService
from utils.config import Config
from utils.logger import setup_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logger(__name__)

# Global services (initialized on startup)
document_service = None
embedding_service = None
ai_service = None
cache_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    global document_service, embedding_service, ai_service, cache_service
    
    logger.info("🚀 Starting HackRX API...")
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        
        cache_service = CacheService()
        embedding_service = EmbeddingService()
        ai_service = AIService()
        document_service = DocumentService()
        
        # Warm up services
        await embedding_service.warmup()
        await ai_service.warmup()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise RuntimeError(f"Startup failed: {e}")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down services...")
    if cache_service:
        await cache_service.cleanup()
    logger.info("✅ Shutdown complete")

# FastAPI app with lifespan management
app = FastAPI(
    title="HackRX Document Q&A API - Modular",
    description="High-performance modular RAG API using Gemini 2.5 Flash Lite",
    version="3.0.0",
    lifespan=lifespan
)

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(
    request: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Main endpoint for document Q&A processing.
    Orchestrates all services to provide answers.
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    logger.info(f"[{request_id}] Processing {len(request.questions)} questions")
    
    try:
        # Validate request
        if not request.documents or not request.questions:
            raise HTTPException(
                status_code=400,
                detail="Both 'documents' URL and 'questions' list are required"
            )
        
        if len(request.questions) > Config.MAX_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Too many questions. Maximum {Config.MAX_QUESTIONS} allowed."
            )
        
        # Check cache first
        cached_answers = await cache_service.get_cached_answers(
            request.documents, request.questions
        )
        
        if all(answer is not None for answer in cached_answers):
            logger.info(f"[{request_id}] All answers from cache - {time.time() - start_time:.2f}s")
            return HackRxResponse(answers=cached_answers)
        
        # Process document
        logger.info(f"[{request_id}] Downloading and processing document...")
        text_content = await document_service.extract_text_from_url(request.documents)
        
        # Create embeddings
        logger.info(f"[{request_id}] Creating document embeddings...")
        chunks, chunk_embeddings = await embedding_service.create_document_embeddings(text_content)
        
        # Get answers from AI service
        logger.info(f"[{request_id}] Generating answers...")
        answers = await ai_service.answer_questions(
            request.questions, chunks, chunk_embeddings
        )
        
        # Cache results
        await cache_service.cache_answers(request.documents, request.questions, answers)
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] ✅ Completed in {total_time:.2f}s")
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        services={
            "document_service": document_service is not None,
            "embedding_service": embedding_service is not None,
            "ai_service": ai_service is not None,
            "cache_service": cache_service is not None,
        },
        timestamp=time.time()
    )

@app.get("/metrics")
async def get_metrics():
    """Performance metrics endpoint"""
    return {
        "cache_stats": await cache_service.get_stats() if cache_service else {},
        "embedding_stats": embedding_service.get_stats() if embedding_service else {},
        "ai_stats": ai_service.get_stats() if ai_service else {},
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    config = Config()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        log_level="info",
        reload=config.DEBUG
    )