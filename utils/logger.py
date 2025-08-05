# utils/logger.py - Logging Configuration
"""
Centralized logging configuration and utilities.
Team member: DevOps/Monitoring Lead
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting across all modules.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level override
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or default to INFO
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create formatter
    formatter = create_formatter()
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def create_formatter() -> logging.Formatter:
    """Create a consistent log formatter"""
    
    # Check if we're in production for different formatting
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    if is_production:
        # JSON format for production (better for log aggregation)
        format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    else:
        # Human-readable format for development
        format_string = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    
    return logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_request_logger(request_id: str) -> logging.LoggerAdapter:
    """
    Get a logger adapter that includes request ID in all log messages.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Logger adapter with request context
    """
    logger = setup_logger("request")
    return logging.LoggerAdapter(logger, {"request_id": request_id})

class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(f"perf.{name}")
        self.start_time = None
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {operation}")
    
    def end_timer(self, operation: str, extra_info: dict = None):
        """End timing and log duration"""
        if self.start_time is None:
            self.logger.warning(f"Timer not started for {operation}")
            return
        
        import time
        duration = time.time() - self.start_time
        
        info_str = ""
        if extra_info:
            info_parts = [f"{k}={v}" for k, v in extra_info.items()]
            info_str = f" ({', '.join(info_parts)})"
        
        self.logger.info(f"Completed {operation} in {duration:.3f}s{info_str}")
        self.start_time = None
        
        return duration

class ServiceLogger:
    """Specialized logger for service classes"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = setup_logger(f"service.{service_name}")
        self.perf_logger = PerformanceLogger(service_name)
    
    def info(self, message: str, **kwargs):
        """Log info message with service context"""
        self._log_with_context("info", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with service context"""
        self._log_with_context("error", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with service context"""
        self._log_with_context("warning", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with service context"""
        self._log_with_context("debug", message, **kwargs)
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with additional context"""
        context_parts = []
        
        if kwargs:
            context_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
        
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""
        full_message = f"[{self.service_name}] {message}{context_str}"
        
        getattr(self.logger, level)(full_message)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.perf_logger.logger.info(
            f"[{self.service_name}] {operation}: {duration:.3f}s ({metrics_str})"
        )

# Global logger instances for common use
api_logger = setup_logger("api")
security_logger = setup_logger("security")
error_logger = setup_logger("error")

def log_api_request(method: str, path: str, status_code: int, duration: float, **kwargs):
    """Log API request with standard format"""
    extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
    api_logger.info(
        f"{method} {path} - {status_code} - {duration:.3f}s {extra_info}".strip()
    )

def log_security_event(event_type: str, details: str, **kwargs):
    """Log security-related events"""
    context = " ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
    security_logger.warning(f"SECURITY: {event_type} - {details} {context}".strip())

def log_error_with_context(error: Exception, context: dict = None):
    """Log error with full context"""
    import traceback
    
    context_str = ""
    if context:
        context_str = " | Context: " + ", ".join([f"{k}={v}" for k, v in context.items()])
    
    error_logger.error(
        f"ERROR: {type(error).__name__}: {str(error)}{context_str}\n"
        f"Traceback: {traceback.format_exc()}"
    )

# Configure third-party loggers to reduce noise
def configure_third_party_loggers():
    """Configure logging levels for third-party libraries"""
    
    # Reduce noise from common libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Keep important warnings
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

# Auto-configure on import
configure_third_party_loggers()