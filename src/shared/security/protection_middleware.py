"""
Comprehensive protection middleware for Project Synapse.

Provides security headers, CORS protection, request validation,
and various security measures for the FastAPI application.
"""

import asyncio
import hashlib
import hmac
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Union, Tuple
from urllib.parse import urlparse
import ipaddress
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..logging_config import get_logger, CorrelationContext, set_correlation_id, set_user_id
from ..metrics_collector import get_metrics_collector
from .secrets_manager import get_secrets_manager


class SecurityConfig:
    """Security configuration settings."""
    
    def __init__(self):
        # CORS settings
        self.cors_enabled = True
        self.cors_allow_origins = ["*"]  # Configure for production
        self.cors_allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.cors_allow_headers = ["*"]
        self.cors_allow_credentials = True
        self.cors_max_age = 86400  # 24 hours
        
        # Security headers
        self.security_headers_enabled = True
        self.hsts_enabled = True
        self.hsts_max_age = 31536000  # 1 year
        self.hsts_include_subdomains = True
        self.content_type_nosniff = True
        self.frame_options = "DENY"
        self.xss_protection = "1; mode=block"
        self.referrer_policy = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        self.csp_enabled = True
        self.csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none';"
        )
        
        # Request validation
        self.request_validation_enabled = True
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_header_size = 8192  # 8KB
        self.max_query_params = 100
        self.max_form_fields = 100
        
        # IP filtering
        self.ip_filtering_enabled = False
        self.allowed_ips: Set[str] = set()
        self.blocked_ips: Set[str] = set()
        self.trusted_proxies: Set[str] = set()
        
        # User-Agent filtering
        self.user_agent_filtering_enabled = True
        self.blocked_user_agents = [
            r".*bot.*",
            r".*crawler.*",
            r".*spider.*",
            r".*scraper.*"
        ]
        self.allowed_user_agents = []  # If set, only these are allowed
        
        # Request signing
        self.request_signing_enabled = False
        self.signature_header = "X-Signature"
        self.timestamp_header = "X-Timestamp"
        self.signature_tolerance = 300  # 5 minutes
        
        # Honeypot protection
        self.honeypot_enabled = True
        self.honeypot_paths = [
            "/admin", "/wp-admin", "/.env", "/config",
            "/phpmyadmin", "/mysql", "/database"
        ]
        
        # Geolocation filtering
        self.geo_filtering_enabled = False
        self.allowed_countries: Set[str] = set()
        self.blocked_countries: Set[str] = set()


class ThreatDetector:
    """Detects various security threats and suspicious patterns."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'threat_detector')
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(?i)(union\s+select)",
            r"(?i)(select\s+.*\s+from)",
            r"(?i)(insert\s+into)",
            r"(?i)(delete\s+from)",
            r"(?i)(drop\s+table)",
            r"(?i)(update\s+.*\s+set)",
            r"(?i)('\s*or\s+'1'\s*=\s*'1)",
            r"(?i)('\s*or\s+1\s*=\s*1)",
            r"(?i)(--\s*$)",
            r"(?i)(/\*.*\*/)"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"(?i)(<script[^>]*>.*?</script>)",
            r"(?i)(<iframe[^>]*>.*?</iframe>)",
            r"(?i)(javascript:)",
            r"(?i)(on\w+\s*=)",
            r"(?i)(<img[^>]*onerror)",
            r"(?i)(<svg[^>]*onload)",
            r"(?i)(eval\s*\()",
            r"(?i)(expression\s*\()"
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\.[\\\/]",
            r"%2e%2e[\\\/]",
            r"\.\.%2f",
            r"%2e%2e%2f"
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"(?i)(;\s*rm\s)",
            r"(?i)(;\s*cat\s)",
            r"(?i)(;\s*ls\s)",
            r"(?i)(;\s*wget\s)",
            r"(?i)(;\s*curl\s)",
            r"(?i)(\|\s*nc\s)",
            r"(?i)(\&\&\s*rm\s)",
            r"(?i)(\|\|\s*rm\s)"
        ]
    
    def detect_sql_injection(self, text: str) -> bool:
        """Detect SQL injection attempts."""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def detect_xss(self, text: str) -> bool:
        """Detect XSS attempts."""
        for pattern in self.xss_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def detect_path_traversal(self, text: str) -> bool:
        """Detect path traversal attempts."""
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def detect_command_injection(self, text: str) -> bool:
        """Detect command injection attempts."""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyze request for security threats."""
        threats = []
        
        # Check URL path
        path = str(request.url.path)
        if self.detect_path_traversal(path):
            threats.append("path_traversal")
        
        # Check query parameters
        for key, value in request.query_params.items():
            combined = f"{key}={value}"
            if self.detect_sql_injection(combined):
                threats.append("sql_injection")
            if self.detect_xss(combined):
                threats.append("xss")
            if self.detect_command_injection(combined):
                threats.append("command_injection")
        
        # Check headers
        for key, value in request.headers.items():
            combined = f"{key}: {value}"
            if self.detect_xss(combined):
                threats.append("xss_header")
            if self.detect_command_injection(combined):
                threats.append("command_injection_header")
        
        return {
            "threats_detected": threats,
            "threat_count": len(threats),
            "risk_level": self._calculate_risk_level(threats)
        }
    
    def _calculate_risk_level(self, threats: List[str]) -> str:
        """Calculate risk level based on detected threats."""
        if not threats:
            return "low"
        elif len(threats) == 1:
            return "medium"
        else:
            return "high"


class RequestValidator:
    """Validates incoming requests for security compliance."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger(__name__, 'request_validator')
    
    async def validate_request(self, request: Request) -> Tuple[bool, Optional[str]]:
        """Validate request against security rules."""
        
        # Check request size
        content_length = int(request.headers.get('content-length', 0))
        if content_length > self.config.max_request_size:
            return False, f"Request too large: {content_length} bytes"
        
        # Check header size
        total_header_size = sum(len(f"{k}: {v}") for k, v in request.headers.items())
        if total_header_size > self.config.max_header_size:
            return False, f"Headers too large: {total_header_size} bytes"
        
        # Check query parameters count
        if len(request.query_params) > self.config.max_query_params:
            return False, f"Too many query parameters: {len(request.query_params)}"
        
        # Check User-Agent
        if self.config.user_agent_filtering_enabled:
            user_agent = request.headers.get('user-agent', '')
            if not self._validate_user_agent(user_agent):
                return False, "Blocked user agent"
        
        # Check IP address
        if self.config.ip_filtering_enabled:
            client_ip = self._get_client_ip(request)
            if not self._validate_ip(client_ip):
                return False, f"Blocked IP address: {client_ip}"
        
        # Check honeypot paths
        if self.config.honeypot_enabled:
            if self._is_honeypot_path(str(request.url.path)):
                return False, "Honeypot triggered"
        
        return True, None
    
    def _validate_user_agent(self, user_agent: str) -> bool:
        """Validate User-Agent header."""
        if not user_agent:
            return False
        
        # Check allowed list first (if configured)
        if self.config.allowed_user_agents:
            for allowed_pattern in self.config.allowed_user_agents:
                if re.match(allowed_pattern, user_agent, re.IGNORECASE):
                    return True
            return False
        
        # Check blocked list
        for blocked_pattern in self.config.blocked_user_agents:
            if re.match(blocked_pattern, user_agent, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_ip(self, ip: str) -> bool:
        """Validate IP address."""
        if not ip or ip == 'unknown':
            return False
        
        # Check blocked IPs
        if ip in self.config.blocked_ips:
            return False
        
        # Check allowed IPs (if configured)
        if self.config.allowed_ips:
            return ip in self.config.allowed_ips
        
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            # Take the first IP in the chain
            ip = forwarded_for.split(',')[0].strip()
            return ip
        
        # Check X-Real-IP header
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        return request.client.host if request.client else 'unknown'
    
    def _is_honeypot_path(self, path: str) -> bool:
        """Check if path is a honeypot."""
        return any(honeypot in path.lower() for honeypot in self.config.honeypot_paths)


class RequestSigner:
    """Handles request signing and verification."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = get_logger(__name__, 'request_signer')
        self.secrets_manager = get_secrets_manager()
    
    async def verify_signature(self, request: Request) -> bool:
        """Verify request signature."""
        if not self.config.request_signing_enabled:
            return True
        
        signature = request.headers.get(self.config.signature_header)
        timestamp = request.headers.get(self.config.timestamp_header)
        
        if not signature or not timestamp:
            return False
        
        # Check timestamp
        try:
            request_time = int(timestamp)
            current_time = int(time.time())
            
            if abs(current_time - request_time) > self.config.signature_tolerance:
                self.logger.warning("Request signature timestamp out of tolerance")
                return False
        except ValueError:
            self.logger.warning("Invalid timestamp in request signature")
            return False
        
        # Verify signature
        try:
            secret_key = await self.secrets_manager.get_secret("request_signing_key")
            if not secret_key:
                self.logger.error("Request signing key not found")
                return False
            
            # Create expected signature
            body = await request.body()
            message = f"{request.method}{request.url.path}{timestamp}{body.decode('utf-8', errors='ignore')}"
            expected_signature = hmac.new(
                secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except Exception as e:
            self.logger.error(f"Error verifying request signature: {e}")
            return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    def __init__(self, app: ASGIApp, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.logger = get_logger(__name__, 'security_headers')
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        if not self.config.security_headers_enabled:
            return response
        
        # HSTS
        if self.config.hsts_enabled:
            hsts_value = f"max-age={self.config.hsts_max_age}"
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value
        
        # Content type options
        if self.config.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Frame options
        if self.config.frame_options:
            response.headers["X-Frame-Options"] = self.config.frame_options
        
        # XSS protection
        if self.config.xss_protection:
            response.headers["X-XSS-Protection"] = self.config.xss_protection
        
        # Referrer policy
        if self.config.referrer_policy:
            response.headers["Referrer-Policy"] = self.config.referrer_policy
        
        # Content Security Policy
        if self.config.csp_enabled and self.config.csp_policy:
            response.headers["Content-Security-Policy"] = self.config.csp_policy
        
        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with enhanced security."""
    
    def __init__(self, app: ASGIApp, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.logger = get_logger(__name__, 'cors_middleware')
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle CORS requests."""
        if not self.config.cors_enabled:
            return await call_next(request)
        
        origin = request.headers.get('origin')
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, origin)
            return response
        
        # Process actual request
        response = await call_next(request)
        self._add_cors_headers(response, origin)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: Optional[str]):
        """Add CORS headers to response."""
        # Check if origin is allowed
        if origin and not self._is_origin_allowed(origin):
            return
        
        # Set allowed origin
        if "*" in self.config.cors_allow_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
        
        # Set other CORS headers
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.config.cors_allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.config.cors_allow_headers)
        response.headers["Access-Control-Max-Age"] = str(self.config.cors_max_age)
        
        if self.config.cors_allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.config.cors_allow_origins:
            return True
        
        return origin in self.config.cors_allow_origins


class ComprehensiveProtectionMiddleware(BaseHTTPMiddleware):
    """Main protection middleware that combines all security measures."""
    
    def __init__(self, app: ASGIApp, config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.config = config or SecurityConfig()
        self.logger = get_logger(__name__, 'protection_middleware')
        self.metrics = get_metrics_collector()
        
        # Initialize components
        self.threat_detector = ThreatDetector()
        self.request_validator = RequestValidator(self.config)
        self.request_signer = RequestSigner(self.config)
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'threats_detected': 0,
            'requests_blocked': 0,
            'validation_failures': 0,
            'signature_failures': 0
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main protection middleware dispatch."""
        start_time = time.time()
        self.stats['requests_processed'] += 1
        
        # Set correlation ID for request tracking
        correlation_id = self._generate_correlation_id()
        set_correlation_id(correlation_id)
        
        try:
            # 1. Request validation
            is_valid, validation_error = await self.request_validator.validate_request(request)
            if not is_valid:
                self.stats['validation_failures'] += 1
                self.stats['requests_blocked'] += 1
                
                self.logger.warning(
                    f"Request validation failed: {validation_error}",
                    operation="request_validation",
                    client_ip=self.request_validator._get_client_ip(request),
                    path=str(request.url.path)
                )
                
                return self._create_error_response(
                    "Request validation failed",
                    status.HTTP_400_BAD_REQUEST,
                    {"validation_error": validation_error}
                )
            
            # 2. Threat detection
            threat_analysis = self.threat_detector.analyze_request(request)
            if threat_analysis['threats_detected']:
                self.stats['threats_detected'] += 1
                
                if threat_analysis['risk_level'] == 'high':
                    self.stats['requests_blocked'] += 1
                    
                    self.logger.warning(
                        f"High-risk threats detected: {threat_analysis['threats_detected']}",
                        operation="threat_detection",
                        client_ip=self.request_validator._get_client_ip(request),
                        path=str(request.url.path),
                        threats=threat_analysis['threats_detected']
                    )
                    
                    return self._create_error_response(
                        "Security threat detected",
                        status.HTTP_403_FORBIDDEN,
                        {"threat_analysis": threat_analysis}
                    )
                else:
                    # Log medium/low risk threats but allow request
                    self.logger.info(
                        f"Threats detected but allowed: {threat_analysis['threats_detected']}",
                        operation="threat_detection",
                        risk_level=threat_analysis['risk_level']
                    )
            
            # 3. Request signature verification
            if self.config.request_signing_enabled:
                signature_valid = await self.request_signer.verify_signature(request)
                if not signature_valid:
                    self.stats['signature_failures'] += 1
                    self.stats['requests_blocked'] += 1
                    
                    self.logger.warning(
                        "Request signature verification failed",
                        operation="signature_verification",
                        client_ip=self.request_validator._get_client_ip(request)
                    )
                    
                    return self._create_error_response(
                        "Invalid request signature",
                        status.HTTP_401_UNAUTHORIZED
                    )
            
            # 4. Process request
            response = await call_next(request)
            
            # 5. Record metrics
            processing_time = time.time() - start_time
            self.metrics.record_histogram(
                'security_middleware_duration_seconds',
                processing_time,
                path=str(request.url.path),
                method=request.method
            )
            
            # 6. Add security context to response
            response.headers['X-Request-ID'] = correlation_id
            response.headers['X-Security-Scan'] = 'passed'
            
            if threat_analysis['threats_detected']:
                response.headers['X-Threats-Detected'] = str(len(threat_analysis['threats_detected']))
            
            return response
        
        except Exception as e:
            self.logger.error(
                f"Error in protection middleware: {e}",
                operation="middleware_error",
                correlation_id=correlation_id
            )
            
            return self._create_error_response(
                "Internal security error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID for request tracking."""
        return hashlib.sha256(
            f"{time.time()}{id(self)}".encode()
        ).hexdigest()[:16]
    
    def _create_error_response(self, message: str, status_code: int, 
                              details: Optional[Dict[str, Any]] = None) -> JSONResponse:
        """Create standardized error response."""
        response_data = {
            "error": message,
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code
        }
        
        if details:
            response_data.update(details)
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protection middleware statistics."""
        return {
            **self.stats,
            'threat_detection_rate': (
                self.stats['threats_detected'] / max(1, self.stats['requests_processed'])
            ),
            'block_rate': (
                self.stats['requests_blocked'] / max(1, self.stats['requests_processed'])
            )
        }


# Factory functions
def create_security_config(
    cors_origins: List[str] = None,
    enable_request_signing: bool = False,
    enable_ip_filtering: bool = False,
    blocked_user_agents: List[str] = None
) -> SecurityConfig:
    """Create a security configuration with custom settings."""
    config = SecurityConfig()
    
    if cors_origins:
        config.cors_allow_origins = cors_origins
    
    if enable_request_signing:
        config.request_signing_enabled = True
    
    if enable_ip_filtering:
        config.ip_filtering_enabled = True
    
    if blocked_user_agents:
        config.blocked_user_agents.extend(blocked_user_agents)
    
    return config


def create_protection_middleware(
    app: ASGIApp,
    config: Optional[SecurityConfig] = None
) -> ComprehensiveProtectionMiddleware:
    """Create protection middleware with optional configuration."""
    return ComprehensiveProtectionMiddleware(app, config)


# Global instances
_security_config: Optional[SecurityConfig] = None
_protection_middleware: Optional[ComprehensiveProtectionMiddleware] = None


def get_security_config() -> SecurityConfig:
    """Get the global security configuration."""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def get_protection_middleware(app: ASGIApp) -> ComprehensiveProtectionMiddleware:
    """Get the global protection middleware instance."""
    global _protection_middleware
    if _protection_middleware is None:
        _protection_middleware = ComprehensiveProtectionMiddleware(app)
    return _protection_middleware