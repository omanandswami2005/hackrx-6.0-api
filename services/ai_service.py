# services/ai_service.py - Gemini AI Service
"""
AI service for generating answers using Gemini 2.5 Flash Lite.
Team member: AI/LLM Lead
"""

import asyncio
import time
import numpy as np
from typing import List, Optional
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
        self.client: Optional[genai.Client] = None
        self.model_name: str = self.config.GEMINI_MODEL
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
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            if not self.config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required")
            
            # Initialize the client as shown in the example
            self.client = genai.Client(api_key=self.config.GEMINI_API_KEY)
            
            logger.info(f"✅ Gemini client initialized for model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {e}")
            raise RuntimeError(f"Gemini client initialization failed: {e}")
    
    async def warmup(self):
        """Warm up the AI model with a test request"""
        try:
            logger.info("Warming up Gemini model...")
            
            test_prompt = "This is a test prompt to warm up the model. Respond with 'OK'."
            
            response = await self._call_gemini_api(
                context="Test context",
                question=test_prompt,
                timeout=10.0
            )
            
            if response and "OK" in response:
                logger.info("✅ Gemini model warmed up successfully")
            else:
                logger.warning(f"⚠️ Gemini warmup returned an unexpected response: {response}")
                
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
        """
        start_time = time.time()
        
        try:
            tasks = [
                self._answer_single_question(question, chunks, chunk_embeddings)
                for question in questions
            ]
            
            semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            answers = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"Question {i} failed: {answer}")
                    final_answers.append("Error processing question. Please try again.")
                    self.stats.error_count += 1
                else:
                    final_answers.append(answer)
                    self.stats.questions_answered += 1
            
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
            # Import the global embedding service
            from main import embedding_service
            
            relevant_chunks = await embedding_service.find_relevant_chunks(
                question, chunks, chunk_embeddings
            )
            
            if not relevant_chunks:
                return "Information not available in the provided document."
            
            context_parts = []
            total_length = 0
            
            for chunk_text, _ in relevant_chunks:
                chunk_length = len(chunk_text)
                if total_length + chunk_length <= self.config.MAX_CONTEXT_LENGTH:
                    context_parts.append(chunk_text)
                    total_length += chunk_length
                else:
                    break
            
            context = " ".join(context_parts)
            
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
        Call Gemini API with optimized prompting and handle streaming response.
        """
        if not self.client:
            raise RuntimeError("Gemini client is not initialized.")

        full_response = ""
        try:
            contents, generation_config = self._create_optimized_prompt_and_config(context, question)
            
            async def stream_generator():
                nonlocal full_response
                # Use the client to call the streaming generation method
                stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config,
                )
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                
                # Update token usage stats from the final chunk
                if hasattr(stream, 'usage_metadata') and stream.usage_metadata:
                    self.stats.total_tokens_used += getattr(stream.usage_metadata, 'total_token_count', 0)

            await asyncio.wait_for(stream_generator(), timeout=timeout)

            if not full_response:
                self.stats.gemini_api_errors += 1
                return "Unable to generate response. Please try again."

            return self._post_process_answer(full_response)

        except asyncio.TimeoutError:
            logger.error(f"Gemini API timeout after {timeout}s")
            self.stats.gemini_api_errors += 1
            return "Response timeout. Please try again with a simpler question."
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            self.stats.gemini_api_errors += 1
            return "Error generating response. Please try again."

    def _create_optimized_prompt_and_config(self, context: str, question: str) -> tuple[types.Content, types.GenerateContentConfig]:
        """Create the prompt and generation configuration for the Gemini API call."""
        question_type = self._detect_question_type(question)
        
        instruction_map = {
            "specific_value": "Extract the exact value, number, percentage, or time period mentioned in the context. Be precise and include units.",
            "coverage": "Explain what is covered or not covered, including any conditions, limits, or exclusions.",
            "process": "Explain the process, steps, or procedure mentioned in the context.",
            "definition": "Provide the definition or explanation as stated in the context.",
            "general": "Answer based on the information provided in the context."
        }
        instruction = instruction_map.get(question_type, instruction_map["general"])
        
        prompt_text = f"""You are an expert at analyzing insurance policy documents. Answer the question based ONLY on the provided context.

CONTEXT:
{context[:self.config.MAX_CONTEXT_LENGTH]}

QUESTION: {question}

INSTRUCTIONS:
- {instruction}
- If the information is not in the context, respond: "Information not available in the provided document."
- Be specific with numbers, percentages, time periods, and amounts.
- Use exact wording from the document when possible.
- Keep your answer concise but complete.
- Do not make assumptions or add information not in the context.

ANSWER:"""
        
        contents = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)],
        )

        generation_config = types.GenerateContentConfig(
            temperature=self.config.AI_TEMPERATURE,
            max_output_tokens=self.config.MAX_OUTPUT_TOKENS,
            top_p=0.8,
            top_k=20,
        )
        
        return contents, generation_config

    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question to optimize prompting"""
        question_lower = question.lower()
        
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
        
        answer = answer.strip()
        
        if answer.lower().startswith(("context:", "question:", "answer:")):
            lines = answer.split('\n')
            answer = '\n'.join(line for line in lines if not line.lower().startswith(("context:", "question:", "answer:")))
        
        if len(answer) > self.config.MAX_OUTPUT_TOKENS * 5:  # Rough character estimate
            answer = answer[:self.config.MAX_OUTPUT_TOKENS * 5]
        
        if answer and self.stats.questions_answered > 0:
            current_total_len = self.stats.average_response_length * (self.stats.questions_answered)
            self.stats.average_response_length = (current_total_len + len(answer)) / (self.stats.questions_answered + 1)
        
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
            "model_name": self.model_name,
            "client_loaded": self.client is not None,
            "api_success_rate": (
                (self.stats.questions_answered) / max(1, self.stats.requests_processed)
            ),
            "average_tokens_per_response": (
                self.stats.total_tokens_used / max(1, self.stats.questions_answered) if self.stats.questions_answered > 0 else 0
            )
        })
        return stats_dict
    
    async def health_check(self) -> bool:
        """Check if AI service is healthy"""
        try:
            if not self.client:
                return False
            
            test_response = await self._call_gemini_api(
                context="Test context for health check",
                question="Respond with 'OK' if you can process this request",
                timeout=10.0
            )
            
            return "ok" in test_response.lower()
            
        except Exception as e:
            logger.error(f"AI service health check failed: {e}")
            return False
