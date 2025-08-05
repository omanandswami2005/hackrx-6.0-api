# services/ai_service.py - Gemini AI Service
"""
AI service for generating answers using Gemini 2.5 Flash Lite.
Team member: AI/LLM Lead
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from google import genai
from google.genai import types

from models.schemas import AIServiceStats
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class AIService:
    """Service for AI-powered question answering using Gemini"""
    
    def __init__(self):
        self.config = Config()
        self.model: Optional[genai.GenerativeModel] = None
        self.stats = AIServiceStats(
            requests_processed=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            error_count=0,
            questions_answered=0,
            total_tokens_used=0,
            average_response_length=0.0,
            gemini_api_errors=0
        )
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model"""
        try:
            if not self.config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required")
            
            # Configure Gemini API
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            
            # Initialize Gemini 2.5 Flash Lite model
            self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
            
            logger.info(f"✅ Gemini model initialized: {self.config.GEMINI_MODEL}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model: {e}")
            raise RuntimeError(f"Gemini model initialization failed: {e}")
    
    async def warmup(self):
        """Warm up the AI model with a test request"""
        try:
            logger.info("Warming up Gemini model...")
            
            test_prompt = "This is a test prompt to warm up the model. Respond with 'OK'."
            
            response = await self._call_gemini_api(
                context="Test context",
                question=test_prompt,
                timeout=5.0
            )
            
            if response:
                logger.info("✅ Gemini model warmed up successfully")
            else:
                logger.warning("⚠️ Gemini warmup returned empty response")
                
        except Exception as e:
            logger.error(f"❌ Gemini model warmup failed: {e}")
            # Don't raise - warmup failure shouldn't stop the service
    
    async def answer_questions(
        self, 
        questions: List[str], 
        chunks: List[str], 
        chunk_embeddings: np.ndarray
    ) -> List[str]:
        """
        Answer multiple questions using the most relevant chunks.
        
        Args:
            questions: List of questions to answer
            chunks: Document text chunks
            chunk_embeddings: Precomputed embeddings for chunks
            
        Returns:
            List of answers corresponding to questions
        """
        start_time = time.time()
        
        try:
            # Process questions concurrently for better performance
            tasks = [
                self._answer_single_question(question, chunks, chunk_embeddings)
                for question in questions
            ]
            
            # Limit concurrency to avoid API rate limits
            semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS // 2) # limit to number of questions = 2
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            answers = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"Question {i} failed: {answer}")
                    final_answers.append("Error processing question. Please try again.")
                    self.stats.error_count += 1
                else:
                    final_answers.append(answer)
                    self.stats.questions_answered += 1
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(questions))
            
            logger.info(f"Answered {len(questions)} questions in {processing_time:.2f}s")
            
            return final_answers
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, 0, success=False)
            logger.error(f"Batch question answering failed: {e}")
            raise RuntimeError(f"Failed to answer questions: {str(e)}")
    
    async def _answer_single_question(
        self, 
        question: str, 
        chunks: List[str], 
        chunk_embeddings: np.ndarray
    ) -> str:
        """Answer a single question using relevant chunks"""
        try:
            # Import here to avoid circular imports
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            embedding_service.model = embedding_service._load_model_sync()  # Quick load for similarity
            
            # Find relevant chunks for this question
            relevant_chunks = await embedding_service.find_relevant_chunks(
                question, chunks, chunk_embeddings
            )
            
            if not relevant_chunks:
                return "Information not available in the provided document."
            
            # Combine relevant chunks as context
            context_parts = []
            total_length = 0
            
            for chunk_text, similarity in relevant_chunks:
                chunk_length = len(chunk_text)
                if total_length + chunk_length <= self.config.MAX_CONTEXT_LENGTH:
                    context_parts.append(chunk_text)
                    total_length += chunk_length
                else:
                    break
            
            context = " ".join(context_parts)
            
            # Get answer from Gemini
            answer = await self._call_gemini_api(context, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Single question answering failed: {e}")
            return "Error processing question. Please try again."
    
    async def _call_gemini_api(
        self, 
        context: str, 
        question: str, 
        timeout: float = 20.0
    ) -> str:
        """
        Call Gemini API with optimized prompting for insurance documents.
        
        Args:
            context: Relevant text context
            question: Question to answer
            timeout: Request timeout in seconds
            
        Returns:
            Generated answer
        """
        try:
            # Optimized prompt for insurance/policy documents
            prompt = self._create_optimized_prompt(context, question)
            
            # Configure generation parameters for speed and accuracy
            generation_config = types.GenerationConfig(
                temperature=self.config.AI_TEMPERATURE,
                max_output_tokens=self.config.MAX_OUTPUT_TOKENS,
                top_p=0.8,  # Slightly focused sampling
                top_k=20,   # Limit token choices for consistency
            )
            
            # Make async API call with timeout
            response = await asyncio.wait_for(
                self._generate_content_async(prompt, generation_config),
                timeout=timeout
            )
            
            if not response or not response.text:
                self.stats.gemini_api_errors += 1
                return "Unable to generate response. Please try again."
            
            # Post-process response
            answer = self._post_process_answer(response.text)
            
            # Update token usage stats (if available)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.stats.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)
            
            return answer
            
        except asyncio.TimeoutError:
            logger.error(f"Gemini API timeout after {timeout}s")
            self.stats.gemini_api_errors += 1
            return "Response timeout. Please try again with a simpler question."
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            self.stats.gemini_api_errors += 1
            return "Error generating response. Please try again."
    
    async def _generate_content_async(self, prompt: str, config: types.GenerationConfig):
        """Generate content asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.model.generate_content(prompt, generation_config=config)
        )
    
    def _create_optimized_prompt(self, context: str, question: str) -> str:
        """Create optimized prompt for insurance/policy documents"""
        
        # Detect question type for better prompting
        question_type = self._detect_question_type(question)
        
        if question_type == "specific_value":
            instruction = """Extract the exact value, number, percentage, or time period mentioned in the context. Be precise and include units."""
        elif question_type == "coverage":
            instruction = """Explain what is covered or not covered, including any conditions, limits, or exclusions."""
        elif question_type == "process":
            instruction = """Explain the process, steps, or procedure mentioned in the context."""
        elif question_type == "definition":
            instruction = """Provide the definition or explanation as stated in the context."""
        else:
            instruction = """Answer based on the information provided in the context."""
        
        prompt = f"""You are an expert at analyzing insurance policy documents. Answer the question based ONLY on the provided context.

CONTEXT:
{context[:self.config.MAX_CONTEXT_LENGTH]}

QUESTION: {question}

INSTRUCTIONS:
- {instruction}
- If the information is not in the context, respond: "Information not available in the provided document."
- Be specific with numbers, percentages, time periods, and amounts
- Use exact wording from the document when possible
- Keep your answer concise but complete
- Do not make assumptions or add information not in the context

ANSWER:"""
        
        return prompt
    
    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question to optimize prompting"""
        question_lower = question.lower()
        
        # Keywords for different question types
        value_keywords = ["how much", "what is the", "percentage", "amount", "period", "time", "days", "months", "years"]
        coverage_keywords = ["cover", "covered", "coverage", "benefit", "eligible", "include", "exclude"]
        process_keywords = ["how to", "process", "procedure", "steps", "apply", "claim"]
        definition_keywords = ["what is", "define", "definition", "meaning", "what does"]
        
        if any(keyword in question_lower for keyword in value_keywords):
            return "specific_value"
        elif any(keyword in question_lower for keyword in coverage_keywords):
            return "coverage"
        elif any(keyword in question_lower for keyword in process_keywords):
            return "process"
        elif any(keyword in question_lower for keyword in definition_keywords):
            return "definition"
        else:
            return "general"
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the AI-generated answer"""
        if not answer:
            return "No response generated."
        
        # Clean up the answer
        answer = answer.strip()
        
        # Remove any potential prompt leakage
        if answer.lower().startswith(("context:", "question:", "answer:")):
            lines = answer.split('\n')
            answer = '\n'.join(line for line in lines if not line.lower().startswith(("context:", "question:", "answer:")))
        
        # Ensure reasonable length
        if len(answer) > self.config.MAX_OUTPUT_TOKENS * 4:  # Rough character estimate
            sentences = answer.split('.')
            answer = '.'.join(sentences[:3]) + '.'  # Keep first 3 sentences
        
        # Update response length stats
        if answer:
            current_avg = self.stats.average_response_length
            count = self.stats.questions_answered
            self.stats.average_response_length = (current_avg * count + len(answer)) / (count + 1)
        
        return answer.strip()
    
    def _update_stats(self, processing_time: float, question_count: int, success: bool = True):
        """Update service statistics"""
        self.stats.requests_processed += 1
        self.stats.total_processing_time += processing_time
        self.stats.average_processing_time = (
            self.stats.total_processing_time / self.stats.requests_processed
        )
        self.stats.last_request_time = time.time()
        
        if not success:
            self.stats.error_count += 1
    
    def get_stats(self) -> dict:
        """Get service statistics"""
        stats_dict = self.stats.dict()
        stats_dict.update({
            "model_name": self.config.GEMINI_MODEL,
            "model_loaded": self.model is not None,
            "api_success_rate": (
                (self.stats.questions_answered) / max(1, self.stats.requests_processed)
            ),
            "average_tokens_per_response": (
                self.stats.total_tokens_used / max(1, self.stats.questions_answered)
            )
        })
        return stats_dict
    
    async def health_check(self) -> bool:
        """Check if AI service is healthy"""
        try:
            if not self.model:
                return False
            
            # Quick health check with simple prompt
            test_response = await self._call_gemini_api(
                context="Test context for health check",
                question="Respond with 'OK' if you can process this request",
                timeout=5.0
            )
            
            return "ok" in test_response.lower()
            
        except Exception as e:
            logger.error(f"AI service health check failed: {e}")
            return False