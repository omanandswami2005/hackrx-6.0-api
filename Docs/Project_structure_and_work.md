# HackRX Modular RAG API - Project Structure

## 📁 Project Structure

```
hackrx-api/
├── main.py                     # Main FastAPI application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Local development setup
├── Procfile                   # Heroku deployment
├── render.yaml                # Render deployment
├── railway.json               # Railway deployment
│
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic models and schemas
│
├── services/
│   ├── __init__.py
│   ├── document_service.py    # PDF processing and text extraction
│   ├── embedding_service.py   # Text embeddings and similarity
│   ├── ai_service.py          # Gemini AI integration
│   └── cache_service.py       # Redis caching service
│
├── utils/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── logger.py              # Logging utilities
│
└── tests/                     # Test files (optional)
    ├── __init__.py
    ├── test_api.py
    ├── test_services.py
    └── test_utils.py
```

## 👥 Team Member Assignments

### 1. **Backend/API Lead**

**Files:** `main.py`, `models/schemas.py`
**Responsibilities:**

* FastAPI application setup and routing
* Request/response validation with Pydantic models
* API endpoint implementation
* Error handling and HTTP status codes
* Health check and metrics endpoints

**Key Features to Implement:**

* Main `/hackrx/run` endpoint orchestration
* Request validation and rate limiting
* Response formatting and error handling
* API documentation and OpenAPI schema

### 2. **Document Processing Lead**

**Files:** `services/document_service.py`
**Responsibilities:**

* PDF download and text extraction
* Text cleaning and preprocessing
* Intelligent text chunking with overlap
* Document validation and error handling

**Key Features to Implement:**

* Async PDF download with timeout handling
* Multi-page PDF text extraction
* Sentence-aware text chunking
* Text cleaning and normalization
* Performance optimization for large documents

### 3. **ML/Embeddings Lead**

**Files:** `services/embedding_service.py`
**Responsibilities:**

* Sentence transformer model management
* Text embedding generation
* Similarity computation and chunk retrieval
* Embedding caching and optimization

**Key Features to Implement:**

* Model loading and warmup
* Batch embedding generation
* Cosine similarity computation
* Top-k chunk retrieval with filtering
* Memory optimization and caching

### 4. **AI/LLM Lead**

**Files:** `services/ai_service.py`
**Responsibilities:**

* Gemini 2.5 Flash Lite integration
* Prompt engineering for insurance documents
* Response generation and post-processing
* AI service monitoring and error handling

**Key Features to Implement:**

* Optimized prompts for different question types
* Async Gemini API calls with timeout
* Response quality improvement
* Token usage tracking
* Concurrent question processing

### 5. **DevOps/Infrastructure Lead**

**Files:** `services/cache_service.py`, `utils/config.py`, `utils/logger.py`, deployment files
**Responsibilities:**

* Redis caching implementation
* Configuration management
* Logging and monitoring
* Deployment setup and optimization

**Key Features to Implement:**

* Redis connection management with fallbacks
* Environment-based configuration
* Structured logging across services
* Docker and cloud deployment configs
* Performance monitoring and metrics

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Python 3.11+
# Redis (optional, for caching)
# Gemini API key
```

### Setup

```bash
# 1. Clone and setup
git clone <repo-url>
cd hackrx-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run locally
uvicorn main:app --reload --port 8000

# 5. Or with Redis using Docker
docker-compose up
```

### Testing

```bash
# Test the API
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is covered?", "What are the limits?"]
  }'
```

## 🎯 Performance Optimizations

### Speed Improvements

1. **Concurrent Processing** : Multiple questions processed simultaneously
2. **Optimized Chunking** : Sentence-aware chunking maintains context
3. **Efficient Embeddings** : Batch processing and caching
4. **Smart Caching** : Redis with in-memory fallback
5. **Async Operations** : Non-blocking I/O throughout

### Accuracy Improvements

1. **Better Prompting** : Question-type-specific prompts
2. **Multiple Chunks** : Top-3 relevant chunks instead of 1
3. **Context Filtering** : Minimum similarity threshold
4. **Response Post-processing** : Answer cleanup and validation

### Reliability Features

1. **Graceful Degradation** : Services work independently
2. **Comprehensive Error Handling** : Detailed error responses
3. **Health Monitoring** : Service health checks and metrics
4. **Timeout Protection** : All external calls have timeouts

## 📊 Expected Performance

### Target Metrics

* **Response Time** : < 15 seconds (vs. original ~30s)
* **Accuracy** : 85%+ on hackathon test cases
* **Reliability** : 99%+ uptime with error handling
* **Concurrency** : Handle 10+ simultaneous requests

### Monitoring

* Real-time metrics at `/metrics` endpoint
* Service health at `/health` endpoint
* Detailed logging for debugging
* Performance statistics per service

## 🚢 Deployment Options

### Quick Deploy (Recommended for Hackathon)

1. **Railway** : `railway up` (with Redis addon)
2. **Render** : Push to GitHub, auto-deploy
3. **Heroku** : `git push heroku main` (with Redis addon)

### Production Deploy

1. **Docker** : Use provided Dockerfile
2. **Kubernetes** : Scale services independently
3. **Cloud Functions** : For serverless deployment

## 🔧 Team Development Tips

### Git Workflow

```bash
# Each team member works on their assigned service
git checkout -b feature/document-service
git checkout -b feature/embedding-service
git checkout -b feature/ai-service
git checkout -b feature/cache-service
git checkout -b feature/api-endpoints
```

### Testing Individual Services

```python
# Test document service
from services.document_service import DocumentService
service = DocumentService()
text = await service.extract_text_from_url("https://example.com/doc.pdf")

# Test embedding service  
from services.embedding_service import EmbeddingService
service = EmbeddingService()
await service.initialize()
chunks, embeddings = await service.create_document_embeddings(text)

# Test AI service
from services.ai_service import AIService
service = AIService()
answers = await service.answer_questions(questions, chunks, embeddings)
```

### Integration Points

* **main.py** orchestrates all services
* **schemas.py** defines data contracts between services
* **config.py** provides shared configuration
* **logger.py** enables consistent logging

This modular structure allows each team member to work independently while ensuring smooth integration and optimal performance for the hackathon submission!
