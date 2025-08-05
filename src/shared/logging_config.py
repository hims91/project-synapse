"""
Comprehensive logging configuration for Project Synapse.

Provides structured logging with correlation IDs, centralized configuration,
and multiple output formats for different environments.
"""

import logging
import logging.config
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
from uuid import uuid4
import traceback
from contextvars import ContextVar
from pathlib import Path
from enum import Enum


class LogLevel(str, Enum):
    """Log levels for filtering."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Context variables for correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation IDs and context to log records."""
    
    def filter(self, record):
        """Add correlation context to log record."""
        record.correlation_id = correlation_id.get() or 'unknown'
        record.user_id = user_id.get() or 'anonymous'
        record.request_id = request_id.get() or 'no-request'
        record.component = getattr(record, 'component', 'unknown')
        record.operation = getattr(record, 'operation', 'unknown')
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra=True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'unknown'),
            'user_id': getattr(record, 'user_id', 'anonymous'),
            'request_id': getattr(record, 'request_id', 'no-request'),
            'component': getattr(record, 'component', 'unknown'),
            'operation': getattr(record, 'operation', 'unknown'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in log_entry and not key.startswith('_'):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format the message
        formatted = super().format(record)
        
        # Add correlation context
        correlation_info = f"[{getattr(record, 'correlation_id', 'unknown')[:8]}]"
        user_info = f"[{getattr(record, 'user_id', 'anon')}]"
        
        return f"{color}{formatted}{reset} {correlation_info} {user_info}"


class SynapseLogger:
    """Enhanced logger with correlation tracking and structured logging."""
    
    def __init__(self, name: str, component: str = None):
        self.logger = logging.getLogger(name)
        self.component = component or name.split('.')[-1]
    
    def _log(self, log_level: int, message: str, operation: str = None, **kwargs):
        """Internal logging method with context."""
        extra = {
            'component': self.component,
            'operation': operation or 'unknown',
            **kwargs
        }
        self.logger.log(log_level, message, extra=extra)
    
    def debug(self, message: str, operation: str = None, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, operation, **kwargs)
    
    def info(self, message: str, operation: str = None, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, operation, **kwargs)
    
    def warning(self, message: str, operation: str = None, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, operation, **kwargs)
    
    def error(self, message: str, operation: str = None, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, operation, **kwargs)
    
    def critical(self, message: str, operation: str = None, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, operation, **kwargs)
    
    def exception(self, message: str, operation: str = None, **kwargs):
        """Log exception with traceback."""
        extra = {
            'component': self.component,
            'operation': operation or 'exception',
            **kwargs
        }
        self.logger.exception(message, extra=extra)


class LoggingConfig:
    """Centralized logging configuration."""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    JSON_FORMAT = JSONFormatter()
    COLORED_FORMAT = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    @classmethod
    def setup_logging(
        cls,
        level: Union[str, int] = logging.INFO,
        format_type: str = 'json',
        log_file: Optional[str] = None,
        console_output: bool = True,
        correlation_tracking: bool = True
    ):
        """
        Setup comprehensive logging configuration.
        
        Args:
            level: Logging level
            format_type: 'json', 'colored', or 'standard'
            log_file: Optional log file path
            console_output: Enable console output
            correlation_tracking: Enable correlation ID tracking
        """
        # Create logs directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add correlation filter if enabled
        correlation_filter = CorrelationFilter() if correlation_tracking else None
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if format_type == 'json':
                console_handler.setFormatter(cls.JSON_FORMAT)
            elif format_type == 'colored':
                console_handler.setFormatter(cls.COLORED_FORMAT)
            else:
                console_handler.setFormatter(logging.Formatter(cls.DEFAULT_FORMAT))
            
            if correlation_filter:
                console_handler.addFilter(correlation_filter)
            
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(cls.JSON_FORMAT)  # Always use JSON for files
            
            if correlation_filter:
                file_handler.addFilter(correlation_filter)
            
            root_logger.addHandler(file_handler)
        
        # Configure specific loggers
        cls._configure_component_loggers()
        
        # Log configuration
        logger = SynapseLogger(__name__, 'logging_config')
        logger.info(
            "Logging system initialized",
            operation="setup_logging",
            level=level,
            format_type=format_type,
            log_file=log_file,
            console_output=console_output,
            correlation_tracking=correlation_tracking
        )
    
    @classmethod
    def _configure_component_loggers(cls):
        """Configure component-specific loggers."""
        # Synapse components
        synapse_loggers = [
            'axon_interface',
            'signal_relay',
            'synaptic_vesicle',
            'dendrites',
            'spinal_cord',
            'shared'
        ]
        
        for component in synapse_loggers:
            logger = logging.getLogger(component)
            logger.setLevel(logging.INFO)
        
        # Third-party loggers (reduce noise)
        third_party_loggers = {
            'uvicorn': logging.WARNING,
            'fastapi': logging.WARNING,
            'httpx': logging.WARNING,
            'aiohttp': logging.WARNING,
            'sqlalchemy': logging.WARNING,
            'alembic': logging.WARNING,
        }
        
        for logger_name, level in third_party_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
    
    @classmethod
    def get_config_dict(
        cls,
        level: str = 'INFO',
        format_type: str = 'json',
        log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get logging configuration as dictionary for dictConfig.
        
        Args:
            level: Logging level
            format_type: Format type
            log_file: Optional log file path
            
        Returns:
            Logging configuration dictionary
        """
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {
                'correlation': {
                    '()': CorrelationFilter,
                }
            },
            'formatters': {
                'json': {
                    '()': JSONFormatter,
                    'include_extra': True
                },
                'colored': {
                    '()': ColoredFormatter,
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': level,
                    'formatter': format_type,
                    'filters': ['correlation'],
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                'axon_interface': {'level': 'INFO'},
                'signal_relay': {'level': 'INFO'},
                'synaptic_vesicle': {'level': 'INFO'},
                'dendrites': {'level': 'INFO'},
                'spinal_cord': {'level': 'INFO'},
                'shared': {'level': 'INFO'},
                'uvicorn': {'level': 'WARNING'},
                'fastapi': {'level': 'WARNING'},
                'httpx': {'level': 'WARNING'},
                'aiohttp': {'level': 'WARNING'},
                'sqlalchemy': {'level': 'WARNING'},
            },
            'root': {
                'level': level,
                'handlers': ['console']
            }
        }
        
        # Add file handler if specified
        if log_file:
            config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'level': level,
                'formatter': 'json',
                'filters': ['correlation'],
                'filename': log_file
            }
            config['root']['handlers'].append('file')
        
        return config


# Context managers for correlation tracking
class CorrelationContext:
    """Context manager for correlation tracking."""
    
    def __init__(self, correlation_id_value: str = None, user_id_value: str = None, request_id_value: str = None):
        self.correlation_id_value = correlation_id_value or str(uuid4())
        self.user_id_value = user_id_value
        self.request_id_value = request_id_value
        self.correlation_token = None
        self.user_token = None
        self.request_token = None
    
    def __enter__(self):
        self.correlation_token = correlation_id.set(self.correlation_id_value)
        if self.user_id_value:
            self.user_token = user_id.set(self.user_id_value)
        if self.request_id_value:
            self.request_token = request_id.set(self.request_id_value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.correlation_token:
            correlation_id.reset(self.correlation_token)
        if self.user_token:
            user_id.reset(self.user_token)
        if self.request_token:
            request_id.reset(self.request_token)


# Utility functions
def get_logger(name: str, component: str = None) -> SynapseLogger:
    """Get a Synapse logger instance."""
    return SynapseLogger(name, component)


def set_correlation_id(correlation_id_value: str):
    """Set correlation ID for current context."""
    correlation_id.set(correlation_id_value)


def set_user_id(user_id_value: str):
    """Set user ID for current context."""
    user_id.set(user_id_value)


def set_request_id(request_id_value: str):
    """Set request ID for current context."""
    request_id.set(request_id_value)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


def get_user_id() -> Optional[str]:
    """Get current user ID."""
    return user_id.get()


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return request_id.get()


# Initialize logging on import
def initialize_logging():
    """Initialize logging with default configuration."""
    environment = os.getenv('ENVIRONMENT', 'development')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    if environment == 'production':
        LoggingConfig.setup_logging(
            level=log_level,
            format_type='json',
            log_file='logs/synapse.log',
            console_output=True,
            correlation_tracking=True
        )
    else:
        LoggingConfig.setup_logging(
            level=log_level,
            format_type='colored',
            console_output=True,
            correlation_tracking=True
        )


# Auto-initialize if not in test environment
if not os.getenv('TESTING'):
    initialize_logging()