"""
Advanced input validation and sanitization system.

Provides comprehensive input validation, sanitization, and security
checks for all types of user input in Project Synapse.
"""

import re
import html
import json
import urllib.parse
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..logging_config import get_logger


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class InputType(str, Enum):
    """Types of input data."""
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    USERNAME = "username"
    PASSWORD = "password"
    JSON = "json"
    HTML = "html"
    SQL = "sql"
    FILENAME = "filename"
    PATH = "path"
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    UUID = "uuid"
    NUMBER = "number"
    DATE = "date"


@dataclass
class ValidationRule:
    """Input validation rule."""
    name: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required: bool = False
    allowed_chars: Optional[Set[str]] = None
    forbidden_chars: Optional[Set[str]] = None
    custom_validator: Optional[callable] = None
    sanitizer: Optional[callable] = None


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputSanitizer:
    """Input sanitization utilities."""
    
    @staticmethod
    def sanitize_html(text: str, allowed_tags: Set[str] = None) -> str:
        """Sanitize HTML content."""
        if not isinstance(text, str):
            return str(text)
        
        # Default allowed tags (very restrictive)
        if allowed_tags is None:
            allowed_tags = {'b', 'i', 'em', 'strong', 'p', 'br'}
        
        # Remove script tags and their content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous attributes
        dangerous_attrs = ['onclick', 'onload', 'onerror', 'onmouseover', 'onfocus', 'onblur']
        for attr in dangerous_attrs:
            text = re.sub(f'{attr}\\s*=\\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        # Remove javascript: and data: URLs
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'data:(?!image/)', '', text, flags=re.IGNORECASE)
        
        # If no tags are allowed, escape everything
        if not allowed_tags:
            return html.escape(text)
        
        # Remove disallowed tags
        def replace_tag(match):
            tag = match.group(1).lower()
            if tag in allowed_tags:
                return match.group(0)
            return ''
        
        text = re.sub(r'<(/?)(\w+)[^>]*>', replace_tag, text)
        
        return text
    
    @staticmethod
    def sanitize_sql(text: str) -> str:
        """Sanitize SQL input."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove SQL comments
        text = re.sub(r'--.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Escape single quotes
        text = text.replace("'", "''")
        
        # Remove dangerous SQL keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'EXEC', 'EXECUTE',
            'UNION', 'INSERT', 'UPDATE', 'MERGE', 'GRANT', 'REVOKE'
        ]
        
        for keyword in dangerous_keywords:
            text = re.sub(f'\\b{keyword}\\b', '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename."""
        if not isinstance(filename, str):
            return str(filename)
        
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove dangerous characters
        dangerous_chars = '<>:"|?*\x00'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_len = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_len] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize file path."""
        if not isinstance(path, str):
            return str(path)
        
        # Remove path traversal attempts
        path = re.sub(r'\.\.[\\/]', '', path)
        path = re.sub(r'[\\/]\.\.', '', path)
        
        # URL decode
        path = urllib.parse.unquote(path)
        
        # Normalize path separators
        path = path.replace('\\', '/')
        
        # Remove multiple slashes
        path = re.sub(r'/+', '/', path)
        
        # Remove leading slash if present
        path = path.lstrip('/')
        
        return path
    
    @staticmethod
    def sanitize_json(data: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
        """Sanitize JSON data recursively."""
        if current_depth > max_depth:
            return None
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitize key
                if isinstance(key, str):
                    sanitized_key = InputSanitizer.sanitize_text(key)
                else:
                    sanitized_key = str(key)
                
                # Sanitize value
                sanitized[sanitized_key] = InputSanitizer.sanitize_json(
                    value, max_depth, current_depth + 1
                )
            return sanitized
        
        elif isinstance(data, list):
            return [
                InputSanitizer.sanitize_json(item, max_depth, current_depth + 1)
                for item in data
            ]
        
        elif isinstance(data, str):
            return InputSanitizer.sanitize_text(data)
        
        else:
            return data
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """General text sanitization."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove control characters except common whitespace
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000]
        
        return text.strip()


class InputValidator:
    """Advanced input validation system."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = get_logger(__name__, 'input_validator')
        self.sanitizer = InputSanitizer()
        
        # Predefined validation patterns
        self.patterns = {
            InputType.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            InputType.URL: r'^https?://[^\s/$.?#].[^\s]*$',
            InputType.PHONE: r'^\+?1?-?\.?\s?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$',
            InputType.USERNAME: r'^[a-zA-Z0-9_-]{3,30}$',
            InputType.UUID: r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            InputType.IP_ADDRESS: r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            InputType.DOMAIN: r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$',
            InputType.NUMBER: r'^-?\d+(\.\d+)?$',
            InputType.DATE: r'^\d{4}-\d{2}-\d{2}$'
        }
        
        # Predefined validation rules
        self.default_rules = {
            InputType.PASSWORD: ValidationRule(
                name="password",
                min_length=8,
                max_length=128,
                custom_validator=self._validate_password_strength
            ),
            InputType.FILENAME: ValidationRule(
                name="filename",
                max_length=255,
                forbidden_chars={'/','\\',':','*','?','"','<','>','|'},
                sanitizer=self.sanitizer.sanitize_filename
            ),
            InputType.PATH: ValidationRule(
                name="path",
                max_length=4096,
                sanitizer=self.sanitizer.sanitize_path
            ),
            InputType.HTML: ValidationRule(
                name="html",
                sanitizer=self.sanitizer.sanitize_html
            ),
            InputType.JSON: ValidationRule(
                name="json",
                custom_validator=self._validate_json,
                sanitizer=self.sanitizer.sanitize_json
            )
        }
    
    def validate(self, value: Any, input_type: InputType, 
                custom_rule: Optional[ValidationRule] = None) -> ValidationResult:
        """Validate input value."""
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        try:
            # Get validation rule
            rule = custom_rule or self.default_rules.get(input_type)
            
            # Convert to string if needed
            if value is not None and not isinstance(value, str):
                value = str(value)
                result.sanitized_value = value
            
            # Check if required
            if rule and rule.required and (value is None or value == ''):
                result.is_valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            
            # Skip validation for empty optional values
            if value is None or value == '':
                return result
            
            # Length validation
            if rule:
                if rule.min_length and len(value) < rule.min_length:
                    result.is_valid = False
                    result.errors.append(f"{rule.name} must be at least {rule.min_length} characters")
                
                if rule.max_length and len(value) > rule.max_length:
                    result.is_valid = False
                    result.errors.append(f"{rule.name} must be at most {rule.max_length} characters")
            
            # Pattern validation
            pattern = rule.pattern if rule else self.patterns.get(input_type)
            if pattern and not re.match(pattern, value):
                result.is_valid = False
                result.errors.append(f"Invalid {input_type.value} format")
            
            # Character validation
            if rule:
                if rule.allowed_chars:
                    invalid_chars = set(value) - rule.allowed_chars
                    if invalid_chars:
                        result.is_valid = False
                        result.errors.append(f"Contains invalid characters: {', '.join(invalid_chars)}")
                
                if rule.forbidden_chars:
                    forbidden_found = set(value) & rule.forbidden_chars
                    if forbidden_found:
                        result.is_valid = False
                        result.errors.append(f"Contains forbidden characters: {', '.join(forbidden_found)}")
            
            # Custom validation
            if rule and rule.custom_validator:
                custom_result = rule.custom_validator(value)
                if not custom_result.get('valid', True):
                    result.is_valid = False
                    result.errors.extend(custom_result.get('errors', []))
                    result.warnings.extend(custom_result.get('warnings', []))
            
            # Sanitization
            if rule and rule.sanitizer:
                result.sanitized_value = rule.sanitizer(value)
            elif input_type == InputType.TEXT:
                result.sanitized_value = self.sanitizer.sanitize_text(value)
            
            # Security checks
            security_result = self._security_check(result.sanitized_value)
            if not security_result['is_safe']:
                if self.validation_level == ValidationLevel.STRICT:
                    result.is_valid = False
                    result.errors.append("Input contains potentially dangerous content")
                else:
                    result.warnings.append("Input may contain suspicious content")
                
                result.metadata['security_threats'] = security_result['threats']
        
        except Exception as e:
            self.logger.error(f"Validation error: {e}", operation="validate_input")
            result.is_valid = False
            result.errors.append("Validation error occurred")
        
        return result
    
    def validate_multiple(self, data: Dict[str, Any], 
                         schema: Dict[str, Tuple[InputType, Optional[ValidationRule]]]) -> Dict[str, ValidationResult]:
        """Validate multiple inputs according to schema."""
        results = {}
        
        for field_name, (input_type, custom_rule) in schema.items():
            value = data.get(field_name)
            results[field_name] = self.validate(value, input_type, custom_rule)
        
        return results
    
    def _validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        if len(password) < 8:
            result['valid'] = False
            result['errors'].append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            if self.validation_level == ValidationLevel.STRICT:
                result['valid'] = False
                result['errors'].append("Password must contain at least one uppercase letter")
            else:
                result['warnings'].append("Password should contain uppercase letters")
        
        if not re.search(r'[a-z]', password):
            if self.validation_level == ValidationLevel.STRICT:
                result['valid'] = False
                result['errors'].append("Password must contain at least one lowercase letter")
            else:
                result['warnings'].append("Password should contain lowercase letters")
        
        if not re.search(r'\d', password):
            if self.validation_level == ValidationLevel.STRICT:
                result['valid'] = False
                result['errors'].append("Password must contain at least one number")
            else:
                result['warnings'].append("Password should contain numbers")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            if self.validation_level == ValidationLevel.STRICT:
                result['valid'] = False
                result['errors'].append("Password must contain at least one special character")
            else:
                result['warnings'].append("Password should contain special characters")
        
        # Check for common patterns
        common_patterns = ['123', 'abc', 'password', 'qwerty']
        for pattern in common_patterns:
            if pattern.lower() in password.lower():
                result['warnings'].append(f"Password contains common pattern: {pattern}")
        
        return result
    
    def _validate_json(self, json_str: str) -> Dict[str, Any]:
        """Validate JSON string."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            data = json.loads(json_str)
            
            # Check depth
            def check_depth(obj, current_depth=0):
                if current_depth > 10:
                    return False
                
                if isinstance(obj, dict):
                    return all(check_depth(v, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, current_depth + 1) for item in obj)
                return True
            
            if not check_depth(data):
                result['valid'] = False
                result['errors'].append("JSON structure too deep")
            
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['errors'].append(f"Invalid JSON: {str(e)}")
        
        return result
    
    def _security_check(self, value: str) -> Dict[str, Any]:
        """Perform basic security checks."""
        if not isinstance(value, str):
            return {'is_safe': True, 'threats': []}
        
        threats = []
        
        # XSS patterns
        xss_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                threats.append('XSS')
                break
        
        # SQL injection patterns
        sql_patterns = [
            r'\b(union|select|insert|update|delete|drop)\b',
            r'(\s|^)(or|and)\s+\d+\s*=\s*\d+',
            r'--\s*$',
            r'/\*.*?\*/'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                threats.append('SQL_INJECTION')
                break
        
        # Path traversal
        if re.search(r'\.\.[\\/]', value):
            threats.append('PATH_TRAVERSAL')
        
        return {
            'is_safe': len(threats) == 0,
            'threats': threats
        }
    
    def create_custom_rule(self, name: str, **kwargs) -> ValidationRule:
        """Create a custom validation rule."""
        return ValidationRule(name=name, **kwargs)


# Utility functions
def validate_email(email: str) -> bool:
    """Quick email validation."""
    validator = InputValidator()
    result = validator.validate(email, InputType.EMAIL)
    return result.is_valid


def validate_url(url: str) -> bool:
    """Quick URL validation."""
    validator = InputValidator()
    result = validator.validate(url, InputType.URL)
    return result.is_valid


def sanitize_user_input(text: str) -> str:
    """Quick text sanitization."""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_text(text)


def validate_and_sanitize(value: str, input_type: InputType) -> Tuple[bool, str, List[str]]:
    """Validate and sanitize input in one call."""
    validator = InputValidator()
    result = validator.validate(value, input_type)
    return result.is_valid, result.sanitized_value, result.errors