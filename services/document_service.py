# services/document_service.py - Document Processing Service
"""
Document download, processing, and text extraction service.
Team member: Document Processing Lead
"""

import aiohttp
import asyncio
from typing import List, Tuple
from io import BytesIO
from pypdf import PdfReader
import re
import time

from models.schemas import DocumentServiceStats
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentService:
    """Service for handling document download and text extraction"""
    
    def __init__(self):
        self.config = Config()
        self.stats = DocumentServiceStats(
            requests_processed=0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            error_count=0,
            documents_processed=0,
            total_pages_processed=0,
            total_text_extracted=0,
            pdf_errors=0
        )
    
    async def extract_text_from_url(self, url: str) -> str:
        """
        Download PDF from URL and extract text content.
        
        Args:
            url: URL to PDF document
            
        Returns:
            Extracted text content
        """
        start_time = time.time()
        
        try:
            # Download PDF content
            pdf_content = await self._download_pdf(url)
            
            # Extract text from PDF
            text_content = await self._extract_text_from_pdf(pdf_content)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=True, text_length=len(text_content))
            
            logger.info(f"Document processed: {len(text_content)} chars in {processing_time:.2f}s")
            return text_content
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=False)
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def _download_pdf(self, url: str) -> bytes:
        """Download PDF content from URL"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.DOWNLOAD_TIMEOUT)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: Failed to download document")
                    
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.config.MAX_FILE_SIZE:
                        raise Exception(f"File too large: {content_length} bytes")
                    
                    content = await response.read()
                    
                    if len(content) > self.config.MAX_FILE_SIZE:
                        raise Exception(f"File too large: {len(content)} bytes")
                    
                    return content
                    
        except asyncio.TimeoutError:
            raise Exception(f"Download timeout after {self.config.DOWNLOAD_TIMEOUT}s")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            # Run PDF processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text_content = await loop.run_in_executor(
                None, self._process_pdf_sync, pdf_content
            )
            
            if not text_content or not text_content.strip():
                raise Exception("No text content extracted from PDF")
            
            return text_content
            
        except Exception as e:
            self.stats.pdf_errors += 1
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    def _process_pdf_sync(self, pdf_content: bytes) -> str:
        """Synchronous PDF processing (runs in thread pool)"""
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            
            if len(reader.pages) == 0:
                raise Exception("PDF has no pages")
            
            # Extract text from all pages
            text_parts = []
            pages_processed = 0
            
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Clean and normalize the text
                        cleaned_text = self._clean_text(page_text)
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                            pages_processed += 1
                            
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i}: {e}")
                    continue
            
            # Update page statistics
            self.stats.total_pages_processed += pages_processed
            
            if not text_parts:
                raise Exception("No readable text found in PDF")
            
            # Combine all text parts
            full_text = ' '.join(text_parts)
            
            # Final text cleaning and validation
            full_text = self._post_process_text(full_text)
            
            return full_text
            
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean individual page text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)  # Space between word and number
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between number and word
        
        return text.strip()
    
    def _post_process_text(self, text: str) -> str:
        """Final text processing and cleanup"""
        if not text:
            return ""
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short fragments (likely extraction artifacts)
        sentences = text.split('.')
        meaningful_sentences = [
            s.strip() for s in sentences 
            if len(s.strip()) > 10  # Keep sentences with meaningful content
        ]
        
        if meaningful_sentences:
            text = '. '.join(meaningful_sentences)
        
        # Ensure text ends properly
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks optimized for RAG.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, create a new chunk
            if current_length + sentence_length > self.config.CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text.strip())
                
                # Keep overlap from previous chunk
                overlap_sentences = current_chunk[-self._calculate_overlap_sentences(current_chunk):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text.strip())
        
        # Filter out very short chunks
        meaningful_chunks = [
            chunk for chunk in chunks 
            if len(chunk.split()) >= 20  # Minimum meaningful chunk size
        ]
        
        return meaningful_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving context"""
        # Simple sentence splitting - can be enhanced with spaCy/nltk if needed
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Filter very short fragments
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        """Calculate number of sentences to keep for overlap"""
        total_words = sum(len(s.split()) for s in sentences)
        overlap_words = min(self.config.CHUNK_OVERLAP, total_words // 2)
        
        # Count sentences from the end until we reach overlap word count
        words_counted = 0
        sentences_to_keep = 0
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if words_counted + sentence_words <= overlap_words:
                words_counted += sentence_words
                sentences_to_keep += 1
            else:
                break
        
        return max(1, sentences_to_keep)  # Keep at least 1 sentence
    
    def _update_stats(self, processing_time: float, success: bool = True, text_length: int = 0):
        """Update service statistics"""
        self.stats.requests_processed += 1
        self.stats.total_processing_time += processing_time
        self.stats.average_processing_time = (
            self.stats.total_processing_time / self.stats.requests_processed
        )
        self.stats.last_request_time = time.time()
        
        if success:
            self.stats.documents_processed += 1
            self.stats.total_text_extracted += text_length
        else:
            self.stats.error_count += 1
    
    def get_stats(self) -> dict:
        """Get service statistics"""
        return self.stats.dict()
    
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            # Simple health check - could be enhanced
            return True
        except Exception:
            return False