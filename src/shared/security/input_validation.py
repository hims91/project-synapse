"""
Input validation and sanitization for Project Synapse.

Provides comprehensive input validation, sanitization, and protection
against common security vulnerabilities like XSS, SQL injection, and
malicious input attacks.
"""

import re
import html
import urllib.parse
import json
import base64
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import bleach
from pydantic import BaseModel, validator, ValidationError

from ..logging_config import get_logger


class ValidationSeverity(str, Enum):
    """Validation error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputType(str, Enum):
    """Types of input for validation."""
    TEXT = "text"
    HTML = "html"
    URL = "url"
    EMAIL = "email"
    JSON = "json"
    SQL = "sql"
    FILENAME = "filename"
    PATH = "path"
    REGEX = "regex"
    BASE64 = "base64"
    HEX = "hex"
    UUID = "uuid"
    IP_ADDRESS = "ip_address"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any
    original_value: Any
    errors: List[str]
    warnings: List[str]
    severity: ValidationSeverity
    input_type: InputType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'sanitized_value': self.sanitized_value,
            'original_value': self.original_value,
            'errors': self.errors,
            'warnings': self.warnings,
            'severity': self.severity.value,
            'input_type': self.input_type.value
        }


class SecurityPatterns:
    """Common security patterns and malicious input detection."""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
        r"(;\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC))",
        r"(\bUNION\s+(ALL\s+)?SELECT)",
        r"(\bINTO\s+(OUTFILE|DUMPFILE))",
        r"(\bLOAD_FILE\s*\()",
        r"(\bSLEEP\s*\()",
        r"(\bBENCHMARK\s*\()"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"onfocus\s*=",
        r"onblur\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<style[^>]*>.*?</style>",
        r"expression\s*\(",
        r"@import",
        r"url\s*\(",
        r"<img[^>]*src\s*=\s*['\"]?javascript:",
        r"<svg[^>]*onload\s*="
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"..%2f",
        r"..%5c",
        r"%252e%252e%252f",
        r"%252e%252e%255c"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]<>]",
        r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|ping|wget|curl|nc|telnet|ssh|ftp)\b",
        r"(\|\s*(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|ping|wget|curl|nc|telnet|ssh|ftp))",
        r"(;\s*(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|ping|wget|curl|nc|telnet|ssh|ftp))",
        r"(`.*`)",
        r"(\$\(.*\))"
    ]
    
    # LDAP injection patterns
    LDAP_INJECTION_PATTERNS = [
        r"[()&|!*]",
        r"\\[0-9a-fA-F]{2}",
        r"\*\)",
        r"\(\|",
        r"\(&"
    ]
    
    # NoSQL injection patterns
    NOSQL_INJECTION_PATTERNS = [
        r"\$where",
        r"\$ne",
        r"\$in",
        r"\$nin",
        r"\$gt",
        r"\$lt",
        r"\$gte",
        r"\$lte",
        r"\$regex",
        r"\$exists",
        r"\$type",
        r"\$mod",
        r"\$all",
        r"\$size",
        r"\$elemMatch"
    ]


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'input_validator')
        
        # Compile regex patterns for performance
        self.sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityPatterns.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityPatterns.XSS_PATTERNS]
        self.path_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityPatterns.PATH_TRAVERSAL_PATTERNS]
        self.cmd_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityPatterns.COMMAND_INJECTION_PATTERNS]
        self.ldap_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityPatterns.LDAP_INJECTION_PATTERNS]
        self.nosql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityPatterns.NOSQL_INJECTION_PATTERNS]
        
        # Allowed HTML tags and attributes for HTML sanitization
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'i', 'b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'img'
        ]
        
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            '*': ['class', 'id']
        }
        
        # Statistics
        self.stats = {
            'validations_performed': 0,
            'threats_detected': 0,
            'sanitizations_performed': 0,
            'errors_found': 0
        }
    
    def validate_input(
        self,
        value: Any,
        input_type: InputType,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        allow_empty: bool = True,
        custom_patterns: Optional[List[str]] = None,
        strict_mode: bool = False
    ) -> ValidationResult:
        """
        Validate and sanitize input based on type and security rules.
        
        Args:
            value: Input value to validate
            input_type: Type of input for validation
            max_length: Maximum allowed length
            min_length: Minimum required length
            allow_empty: Whether empty values are allowed
            custom_patterns: Additional regex patterns to check
            strict_mode: Enable strict validation mode
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        self.stats['validations_performed'] += 1
        
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW
        sanitized_value = value
        
        try:
            # Convert to string for validation
            str_value = str(value) if value is not None else ""
            
            # Check if empty
            if not str_value.strip():
                if not allow_empty:
                    errors.append("Empty value not allowed")
                    severity = ValidationSeverity.MEDIUM
                return ValidationResult(
                    is_valid=len(errors) == 0,
                    sanitized_value="" if allow_empty else None,
                    original_value=value,
                    errors=errors,
                    warnings=warnings,
                    severity=severity,
                    input_type=input_type
                )
            
            # Length validation
            if max_length and len(str_value) > max_length:
                errors.append(f"Value exceeds maximum length of {max_length}")
                severity = ValidationSeverity.MEDIUM
            
            if min_length and len(str_value) < min_length:
                errors.append(f"Value below minimum length of {min_length}")
                severity = ValidationSeverity.MEDIUM
            
            # Type-specific validation
            if input_type == InputType.TEXT:
                sanitized_value, type_errors, type_warnings = self._validate_text(str_value, strict_mode)
            elif input_type == InputType.HTML:
                sanitized_value, type_errors, type_warnings = self._validate_html(str_value, strict_mode)
            elif input_type == InputType.URL:
                sanitized_value, type_errors, type_warnings = self._validate_url(str_value, strict_mode)
            elif input_type == InputType.EMAIL:
                sanitized_value, type_errors, type_warnings = self._validate_email(str_value, strict_mode)
            elif input_type == InputType.JSON:
                sanitized_value, type_errors, type_warnings = self._validate_json(str_value, strict_mode)
            elif input_type == InputType.SQL:
                sanitized_value, type_errors, type_warnings = self._validate_sql(str_value, strict_mode)
            elif input_type == InputType.FILENAME:
                sanitized_value, type_errors, type_warnings = self._validate_filename(str_value, strict_mode)
            elif input_type == InputType.PATH:
                sanitized_value, type_errors, type_warnings = self._validate_path(str_value, strict_mode)
            elif input_type == InputType.UUID:
                sanitized_value, type_errors, type_warnings = self._validate_uuid(str_value, strict_mode)
            elif input_type == InputType.IP_ADDRESS:
                sanitized_value, type_errors, type_warnings = self._validate_ip_address(str_value, strict_mode)
            else:
                sanitized_value, type_errors, type_warnings = self._validate_generic(str_value, strict_mode)
            
            errors.extend(type_errors)
            warnings.extend(type_warnings)
            
            # Custom pattern validation
            if custom_patterns:
                for pattern in custom_patterns:
                    if re.search(pattern, str_value, re.IGNORECASE):
                        errors.append(f"Input matches restricted pattern: {pattern}")
                        severity = ValidationSeverity.HIGH
            
            # Security threat detection
            threat_level = self._detect_security_threats(str_value)
            if threat_level > ValidationSeverity.LOW:
                severity = max(severity, threat_level)
            
            # Update statistics
            if errors:
                self.stats['errors_found'] += 1
            if sanitized_value != str_value:
                self.stats['sanitizations_performed'] += 1
            if severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]:
                self.stats['threats_detected'] += 1
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_value=sanitized_value,
                original_value=value,
                errors=errors,
                warnings=warnings,
                severity=severity,
                input_type=input_type
            )
        
        except Exception as e:
            self.logger.error(f"Validation error: {e}", operation="validate_input")
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                original_value=value,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                severity=ValidationSeverity.CRITICAL,
                input_type=input_type
            )
    
    def _validate_text(self, value: str, strict_mode: bool) -> tuple:
        """Validate plain text input."""
        errors = []
        warnings = []
        sanitized = value
        
        # Remove null bytes
        if '\x00' in sanitized:
            sanitized = sanitized.replace('\x00', '')
            warnings.append("Null bytes removed")
        
        # Check for control characters
        control_chars = [c for c in sanitized if ord(c) < 32 and c not in ['\t', '\n', '\r']]
        if control_chars:
            sanitized = ''.join(c for c in sanitized if ord(c) >= 32 or c in ['\t', '\n', '\r'])
            warnings.append("Control characters removed")
        
        # HTML entity encoding for safety
        if strict_mode:
            sanitized = html.escape(sanitized)
        
        return sanitized, errors, warnings
    
    def _validate_html(self, value: str, strict_mode: bool) -> tuple:
        """Validate and sanitize HTML input."""
        errors = []
        warnings = []
        
        try:
            # Use bleach to sanitize HTML
            sanitized = bleach.clean(
                value,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
            
            if sanitized != value:
                warnings.append("HTML content sanitized")
            
            return sanitized, errors, warnings
        
        except Exception as e:
            errors.append(f"HTML validation error: {str(e)}")
            return html.escape(value), errors, warnings
    
    def _validate_url(self, value: str, strict_mode: bool) -> tuple:
        """Validate URL input."""
        errors = []
        warnings = []
        sanitized = value.strip()
        
        # Basic URL format validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(sanitized):
            errors.append("Invalid URL format")
        
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'vbscript:', 'data:', 'file:']
        for protocol in dangerous_protocols:
            if sanitized.lower().startswith(protocol):
                errors.append(f"Dangerous protocol detected: {protocol}")
        
        # URL encode special characters
        try:
            parsed = urllib.parse.urlparse(sanitized)
            if parsed.scheme and parsed.netloc:
                sanitized = urllib.parse.urlunparse(parsed)
        except Exception:
            errors.append("URL parsing failed")
        
        return sanitized, errors, warnings
    
    def _validate_email(self, value: str, strict_mode: bool) -> tuple:
        """Validate email address."""
        errors = []
        warnings = []
        sanitized = value.strip().lower()
        
        # Basic email validation
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        if not email_pattern.match(sanitized):
            errors.append("Invalid email format")
        
        # Check for suspicious patterns
        if '..' in sanitized or sanitized.startswith('.') or sanitized.endswith('.'):
            errors.append("Invalid email format: consecutive dots or leading/trailing dots")
        
        return sanitized, errors, warnings
    
    def _validate_json(self, value: str, strict_mode: bool) -> tuple:
        """Validate JSON input."""
        errors = []
        warnings = []
        sanitized = value.strip()
        
        try:
            # Parse JSON to validate format
            parsed = json.loads(sanitized)
            
            # Re-serialize to normalize format
            sanitized = json.dumps(parsed, separators=(',', ':'))
            
            # Check for dangerous content in JSON
            json_str = json.dumps(parsed)
            if self._contains_dangerous_content(json_str):
                warnings.append("Potentially dangerous content in JSON")
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
        
        return sanitized, errors, warnings
    
    def _validate_sql(self, value: str, strict_mode: bool) -> tuple:
        """Validate SQL input (for cases where SQL is expected)."""
        errors = []
        warnings = []
        sanitized = value.strip()
        
        # In strict mode, escape SQL special characters
        if strict_mode:
            sanitized = sanitized.replace("'", "''")
            sanitized = sanitized.replace('"', '""')
            sanitized = sanitized.replace('\\', '\\\\')
        
        # Check for dangerous SQL patterns
        for pattern in self.sql_patterns:
            if pattern.search(sanitized):
                if strict_mode:
                    errors.append("SQL injection pattern detected")
                else:
                    warnings.append("Potential SQL injection pattern detected")
        
        return sanitized, errors, warnings
    
    def _validate_filename(self, value: str, strict_mode: bool) -> tuple:
        """Validate filename input."""
        errors = []
        warnings = []
        sanitized = value.strip()
        
        # Remove path separators
        sanitized = sanitized.replace('/', '_').replace('\\', '_')
        
        # Remove dangerous characters
        dangerous_chars = '<>:"|?*'
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '_')
                warnings.append(f"Dangerous character '{char}' replaced")
        
        # Check for reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        if sanitized.upper() in reserved_names:
            sanitized = f"_{sanitized}"
            warnings.append("Reserved filename modified")
        
        # Length check
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
            warnings.append("Filename truncated to 255 characters")
        
        return sanitized, errors, warnings
    
    def _validate_path(self, value: str, strict_mode: bool) -> tuple:
        """Validate file path input."""
        errors = []
        warnings = []
        sanitized = value.strip()
        
        # Check for path traversal attempts
        for pattern in self.path_patterns:
            if pattern.search(sanitized):
                errors.append("Path traversal attempt detected")
        
        # Normalize path separators
        sanitized = sanitized.replace('\\', '/')
        
        # Remove double slashes
        while '//' in sanitized:
            sanitized = sanitized.replace('//', '/')
        
        # Check for absolute paths in strict mode
        if strict_mode and sanitized.startswith('/'):
            errors.append("Absolute paths not allowed in strict mode")
        
        return sanitized, errors, warnings
    
    def _validate_uuid(self, value: str, strict_mode: bool) -> tuple:
        """Validate UUID input."""
        errors = []
        warnings = []
        sanitized = value.strip().lower()
        
        # UUID pattern validation
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        )
        
        if not uuid_pattern.match(sanitized):
            errors.append("Invalid UUID format")
        
        return sanitized, errors, warnings
    
    def _validate_ip_address(self, value: str, strict_mode: bool) -> tuple:
        """Validate IP address input."""
        errors = []
        warnings = []
        sanitized = value.strip()
        
        # IPv4 pattern
        ipv4_pattern = re.compile(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        )
        
        # IPv6 pattern (simplified)
        ipv6_pattern = re.compile(
            r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        )
        
        if not (ipv4_pattern.match(sanitized) or ipv6_pattern.match(sanitized)):
            errors.append("Invalid IP address format")
        
        # Check for private/reserved IP ranges
        if ipv4_pattern.match(sanitized):
            parts = [int(x) for x in sanitized.split('.')]
            if (parts[0] == 10 or 
                (parts[0] == 172 and 16 <= parts[1] <= 31) or
                (parts[0] == 192 and parts[1] == 168) or
                parts[0] == 127):
                warnings.append("Private/reserved IP address")
        
        return sanitized, errors, warnings
    
    def _validate_generic(self, value: str, strict_mode: bool) -> tuple:
        """Generic validation for unknown input types."""
        errors = []
        warnings = []
        sanitized = value
        
        # Basic sanitization
        sanitized = html.escape(sanitized)
        
        if sanitized != value:
            warnings.append("HTML entities encoded")
        
        return sanitized, errors, warnings
    
    def _detect_security_threats(self, value: str) -> ValidationSeverity:
        """Detect security threats in input."""
        max_severity = ValidationSeverity.LOW
        
        # SQL injection detection
        for pattern in self.sql_patterns:
            if pattern.search(value):
                max_severity = max(max_severity, ValidationSeverity.HIGH)
                self.logger.warning(f"SQL injection pattern detected", operation="threat_detection")
        
        # XSS detection
        for pattern in self.xss_patterns:
            if pattern.search(value):
                max_severity = max(max_severity, ValidationSeverity.HIGH)
                self.logger.warning(f"XSS pattern detected", operation="threat_detection")
        
        # Command injection detection
        for pattern in self.cmd_patterns:
            if pattern.search(value):
                max_severity = max(max_severity, ValidationSeverity.CRITICAL)
                self.logger.warning(f"Command injection pattern detected", operation="threat_detection")
        
        # Path traversal detection
        for pattern in self.path_patterns:
            if pattern.search(value):
                max_severity = max(max_severity, ValidationSeverity.HIGH)
                self.logger.warning(f"Path traversal pattern detected", operation="threat_detection")
        
        # LDAP injection detection
        for pattern in self.ldap_patterns:
            if pattern.search(value):
                max_severity = max(max_severity, ValidationSeverity.MEDIUM)
                self.logger.warning(f"LDAP injection pattern detected", operation="threat_detection")
        
        # NoSQL injection detection
        for pattern in self.nosql_patterns:
            if pattern.search(value):
                max_severity = max(max_severity, ValidationSeverity.MEDIUM)
                self.logger.warning(f"NoSQL injection pattern detected", operation="threat_detection")
        
        return max_severity
    
    def _contains_dangerous_content(self, content: str) -> bool:
        """Check if content contains dangerous patterns."""
        dangerous_keywords = [
            'eval', 'exec', 'system', 'shell_exec', 'passthru', 'file_get_contents',
            'file_put_contents', 'fopen', 'fwrite', 'include', 'require', 'import',
            '__import__', 'subprocess', 'os.system', 'os.popen'
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in dangerous_keywords)
    
    def validate_dict(self, data: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """Validate a dictionary of values according to rules."""
        results = {}
        
        for field, value in data.items():
            if field in validation_rules:
                rules = validation_rules[field]
                input_type = InputType(rules.get('type', InputType.TEXT))
                
                result = self.validate_input(
                    value=value,
                    input_type=input_type,
                    max_length=rules.get('max_length'),
                    min_length=rules.get('min_length'),
                    allow_empty=rules.get('allow_empty', True),
                    custom_patterns=rules.get('custom_patterns'),
                    strict_mode=rules.get('strict_mode', False)
                )
                
                results[field] = result
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.stats.copy()


# Global validator instance
_input_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get the global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


# Convenience functions
def validate_input(value: Any, input_type: InputType, **kwargs) -> ValidationResult:
    """Validate input using the global validator."""
    validator = get_input_validator()
    return validator.validate_input(value, input_type, **kwargs)


def sanitize_html(html_content: str) -> str:
    """Sanitize HTML content."""
    result = validate_input(html_content, InputType.HTML)
    return result.sanitized_value


def validate_url(url: str) -> ValidationResult:
    """Validate URL."""
    return validate_input(url, InputType.URL)


def validate_email(email: str) -> ValidationResult:
    """Validate email address."""
    return validate_input(email, InputType.EMAIL)


def validate_json_input(json_str: str) -> ValidationResult:
    """Validate JSON input."""
    return validate_input(json_str, InputType.JSON)