"""
Webhook validation utilities.

Provides validation for webhook endpoints, events, and delivery configurations.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import logging

from .models import WebhookEventType, WebhookEndpoint, WebhookEvent

logger = logging.getLogger(__name__)


class WebhookValidator:
    """Validates webhook configurations and data."""
    
    @staticmethod
    def validate_endpoint(endpoint: WebhookEndpoint) -> Tuple[bool, List[str]]:
        """
        Validate webhook endpoint configuration.
        
        Args:
            endpoint: Webhook endpoint to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate URL
        url_valid, url_error = WebhookValidator.validate_url(str(endpoint.url))
        if not url_valid:
            errors.append(f"Invalid URL: {url_error}")
        
        # Validate name
        if not endpoint.name or len(endpoint.name.strip()) == 0:
            errors.append("Name is required")
        elif len(endpoint.name) > 100:
            errors.append("Name must be 100 characters or less")
        
        # Validate description
        if endpoint.description and len(endpoint.description) > 500:
            errors.append("Description must be 500 characters or less")
        
        # Validate event types
        if endpoint.event_types:
            invalid_types = [
                et for et in endpoint.event_types 
                if et not in WebhookEventType.__members__.values()
            ]
            if invalid_types:
                errors.append(f"Invalid event types: {invalid_types}")
        
        # Validate timeout
        if endpoint.timeout_seconds < 1 or endpoint.timeout_seconds > 300:
            errors.append("Timeout must be between 1 and 300 seconds")
        
        # Validate max retries
        if endpoint.max_retries < 0 or endpoint.max_retries > 10:
            errors.append("Max retries must be between 0 and 10")
        
        # Validate retry delay
        if endpoint.retry_delay_seconds < 1:
            errors.append("Retry delay must be at least 1 second")
        
        # Validate headers
        header_errors = WebhookValidator.validate_headers(endpoint.headers)
        errors.extend(header_errors)
        
        # Validate event filters
        filter_errors = WebhookValidator.validate_event_filters(endpoint.event_filters)
        errors.extend(filter_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook URL.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
            
            # Must be HTTP or HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False, "URL must use HTTP or HTTPS protocol"
            
            # Must have a hostname
            if not parsed.hostname:
                return False, "URL must have a valid hostname"
            
            # Validate hostname format
            hostname = parsed.hostname.lower()
            
            # Basic hostname validation
            if not re.match(r'^[a-z0-9.-]+$', hostname):
                return False, "Invalid hostname format"
            
            # Check for valid TLD (basic check)
            if '.' not in hostname and hostname != 'localhost':
                return False, "Hostname must have a valid domain"
            
            # Validate port if specified
            if parsed.port:
                if parsed.port < 1 or parsed.port > 65535:
                    return False, "Port must be between 1 and 65535"
            
            # URL length check
            if len(url) > 2048:
                return False, "URL must be 2048 characters or less"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"
    
    @staticmethod
    def validate_headers(headers: Dict[str, str]) -> List[str]:
        """
        Validate webhook headers.
        
        Args:
            headers: Headers dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if not isinstance(headers, dict):
            errors.append("Headers must be a dictionary")
            return errors
        
        # Check header count
        if len(headers) > 20:
            errors.append("Maximum 20 custom headers allowed")
        
        for key, value in headers.items():
            # Validate header name
            if not isinstance(key, str):
                errors.append(f"Header name must be string: {key}")
                continue
            
            if not re.match(r'^[a-zA-Z0-9-_]+$', key):
                errors.append(f"Invalid header name format: {key}")
                continue
            
            if len(key) > 100:
                errors.append(f"Header name too long: {key}")
                continue
            
            # Validate header value
            if not isinstance(value, str):
                errors.append(f"Header value must be string: {key}")
                continue
            
            if len(value) > 1000:
                errors.append(f"Header value too long: {key}")
                continue
            
            # Check for control characters
            if any(ord(c) < 32 and c not in ['\t', '\n', '\r'] for c in value):
                errors.append(f"Header value contains invalid characters: {key}")
        
        return errors
    
    @staticmethod
    def validate_event_filters(filters: Dict[str, Any]) -> List[str]:
        """
        Validate event filters.
        
        Args:
            filters: Event filters dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if not isinstance(filters, dict):
            errors.append("Event filters must be a dictionary")
            return errors
        
        # Check filter count
        if len(filters) > 10:
            errors.append("Maximum 10 event filters allowed")
        
        for key, value in filters.items():
            # Validate filter key
            if not isinstance(key, str):
                errors.append(f"Filter key must be string: {key}")
                continue
            
            if not re.match(r'^[a-zA-Z0-9_.-]+$', key):
                errors.append(f"Invalid filter key format: {key}")
                continue
            
            if len(key) > 50:
                errors.append(f"Filter key too long: {key}")
                continue
            
            # Validate filter value
            if isinstance(value, (str, int, float, bool)):
                # Simple values are OK
                if isinstance(value, str) and len(value) > 200:
                    errors.append(f"Filter value too long: {key}")
            elif isinstance(value, list):
                # List values are OK
                if len(value) > 20:
                    errors.append(f"Filter list too long: {key}")
                for item in value:
                    if not isinstance(item, (str, int, float, bool)):
                        errors.append(f"Invalid filter list item type: {key}")
                        break
            else:
                errors.append(f"Invalid filter value type: {key}")
        
        return errors
    
    @staticmethod
    def validate_event(event: WebhookEvent) -> Tuple[bool, List[str]]:
        """
        Validate webhook event.
        
        Args:
            event: Webhook event to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate event type
        if event.event_type not in WebhookEventType.__members__.values():
            errors.append(f"Invalid event type: {event.event_type}")
        
        # Validate data payload size
        try:
            import json
            payload_size = len(json.dumps(event.data))
            if payload_size > 1024 * 1024:  # 1MB limit
                errors.append("Event data payload too large (max 1MB)")
        except Exception:
            errors.append("Event data is not JSON serializable")
        
        # Validate metadata size
        try:
            import json
            metadata_size = len(json.dumps(event.metadata))
            if metadata_size > 10 * 1024:  # 10KB limit
                errors.append("Event metadata too large (max 10KB)")
        except Exception:
            errors.append("Event metadata is not JSON serializable")
        
        # Validate user_id format if provided
        if event.user_id:
            if not isinstance(event.user_id, str):
                errors.append("User ID must be a string")
            elif len(event.user_id) > 100:
                errors.append("User ID too long (max 100 characters)")
        
        # Validate source
        if not event.source or len(event.source) > 50:
            errors.append("Source must be provided and max 50 characters")
        
        # Validate version
        if not event.version or not re.match(r'^\d+\.\d+$', event.version):
            errors.append("Version must be in format 'X.Y'")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_delivery_config(
        timeout_seconds: int,
        max_retries: int,
        retry_delay_seconds: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate delivery configuration.
        
        Args:
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            retry_delay_seconds: Initial retry delay
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if timeout_seconds < 1 or timeout_seconds > 300:
            errors.append("Timeout must be between 1 and 300 seconds")
        
        if max_retries < 0 or max_retries > 10:
            errors.append("Max retries must be between 0 and 10")
        
        if retry_delay_seconds < 1 or retry_delay_seconds > 3600:
            errors.append("Retry delay must be between 1 and 3600 seconds")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_event_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize event data for webhook delivery.
        
        Args:
            data: Event data dictionary
            
        Returns:
            Sanitized data dictionary
        """
        def sanitize_value(value):
            if isinstance(value, str):
                # Remove control characters except tab, newline, carriage return
                return ''.join(
                    char for char in value 
                    if ord(char) >= 32 or char in ['\t', '\n', '\r']
                )
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            else:
                return value
        
        return sanitize_value(data)
    
    @staticmethod
    def validate_webhook_response(
        status_code: int,
        headers: Dict[str, str],
        body: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook response.
        
        Args:
            status_code: HTTP status code
            headers: Response headers
            body: Response body
            
        Returns:
            Tuple of (is_success, error_message)
        """
        # Success status codes
        if 200 <= status_code < 300:
            return True, None
        
        # Client errors (4xx) - don't retry
        if 400 <= status_code < 500:
            return False, f"Client error: {status_code}"
        
        # Server errors (5xx) - can retry
        if 500 <= status_code < 600:
            return False, f"Server error: {status_code}"
        
        # Other status codes
        return False, f"Unexpected status code: {status_code}"
    
    @staticmethod
    def extract_error_details(
        status_code: Optional[int],
        response_body: Optional[str],
        error_message: Optional[str]
    ) -> str:
        """
        Extract meaningful error details from response.
        
        Args:
            status_code: HTTP status code
            response_body: Response body
            error_message: Error message
            
        Returns:
            Formatted error details
        """
        details = []
        
        if status_code:
            details.append(f"Status: {status_code}")
        
        if error_message:
            details.append(f"Error: {error_message}")
        
        if response_body:
            # Truncate long response bodies
            body = response_body[:500]
            if len(response_body) > 500:
                body += "... (truncated)"
            details.append(f"Response: {body}")
        
        return " | ".join(details) if details else "Unknown error"