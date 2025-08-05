"""
Webhook security utilities.

Provides security features for webhook delivery including signature generation,
verification, and secure header management.
"""

import hmac
import hashlib
import secrets
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WebhookSecurity:
    """Handles webhook security operations."""
    
    @staticmethod
    def generate_secret() -> str:
        """
        Generate a secure webhook secret.
        
        Returns:
            Base64-encoded random secret
        """
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_signature(
        payload: str, 
        secret: str, 
        algorithm: str = "sha256"
    ) -> str:
        """
        Generate HMAC signature for webhook payload.
        
        Args:
            payload: Webhook payload as string
            secret: Webhook secret
            algorithm: Hash algorithm (sha256, sha1)
            
        Returns:
            Hex-encoded signature
        """
        if algorithm == "sha256":
            hash_func = hashlib.sha256
        elif algorithm == "sha1":
            hash_func = hashlib.sha1
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hash_func
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def verify_signature(
        payload: str,
        signature: str,
        secret: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Webhook payload as string
            signature: Received signature (hex-encoded)
            secret: Webhook secret
            algorithm: Hash algorithm
            
        Returns:
            True if signature is valid
        """
        try:
            expected_signature = WebhookSecurity.generate_signature(
                payload, secret, algorithm
            )
            
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False
    
    @staticmethod
    def create_signature_headers(
        payload: str,
        secret: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create signature headers for webhook delivery.
        
        Args:
            payload: Webhook payload as string
            secret: Webhook secret
            timestamp: Optional timestamp for signature
            
        Returns:
            Dictionary of headers to include in request
        """
        headers = {}
        
        # Generate SHA256 signature
        sha256_signature = WebhookSecurity.generate_signature(payload, secret, "sha256")
        headers["X-Synapse-Signature-256"] = f"sha256={sha256_signature}"
        
        # Generate SHA1 signature for compatibility
        sha1_signature = WebhookSecurity.generate_signature(payload, secret, "sha1")
        headers["X-Synapse-Signature"] = f"sha1={sha1_signature}"
        
        # Add timestamp if provided
        if timestamp:
            headers["X-Synapse-Timestamp"] = timestamp
        
        return headers
    
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook URL for security.
        
        Args:
            url: Webhook URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            
            # Must be HTTP or HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False, "URL must use HTTP or HTTPS protocol"
            
            # Must have a hostname
            if not parsed.hostname:
                return False, "URL must have a valid hostname"
            
            # Block localhost and private IPs in production
            hostname = parsed.hostname.lower()
            
            # Block localhost
            if hostname in ['localhost', '127.0.0.1', '::1']:
                return False, "Localhost URLs are not allowed"
            
            # Block private IP ranges (basic check)
            if hostname.startswith('192.168.') or hostname.startswith('10.'):
                return False, "Private IP addresses are not allowed"
            
            if hostname.startswith('172.'):
                # Check if it's in 172.16.0.0/12 range
                parts = hostname.split('.')
                if len(parts) >= 2:
                    try:
                        second_octet = int(parts[1])
                        if 16 <= second_octet <= 31:
                            return False, "Private IP addresses are not allowed"
                    except ValueError:
                        pass
            
            # Block common internal domains
            blocked_domains = [
                'internal', 'local', 'intranet', 'corp', 'lan'
            ]
            
            for blocked in blocked_domains:
                if blocked in hostname:
                    return False, f"Domain containing '{blocked}' is not allowed"
            
            # Must use HTTPS in production (can be relaxed for development)
            # if parsed.scheme != 'https':
            #     return False, "HTTPS is required for webhook URLs"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"
    
    @staticmethod
    def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize webhook headers for security.
        
        Args:
            headers: Headers dictionary
            
        Returns:
            Sanitized headers dictionary
        """
        sanitized = {}
        
        # Allowed header prefixes
        allowed_prefixes = [
            'x-', 'content-', 'accept', 'user-agent', 'authorization'
        ]
        
        # Blocked headers
        blocked_headers = [
            'host', 'connection', 'transfer-encoding', 'upgrade',
            'proxy-', 'sec-', 'x-forwarded-', 'x-real-ip'
        ]
        
        for key, value in headers.items():
            key_lower = key.lower()
            
            # Check if header is blocked
            if any(key_lower.startswith(blocked) for blocked in blocked_headers):
                logger.warning(f"Blocked header: {key}")
                continue
            
            # Check if header is allowed
            if any(key_lower.startswith(prefix) for prefix in allowed_prefixes):
                # Sanitize value (remove control characters)
                sanitized_value = ''.join(
                    char for char in value 
                    if ord(char) >= 32 or char in ['\t', '\n', '\r']
                )
                sanitized[key] = sanitized_value[:1000]  # Limit length
        
        return sanitized
    
    @staticmethod
    def create_delivery_id() -> str:
        """
        Create a unique delivery ID for tracking.
        
        Returns:
            Unique delivery identifier
        """
        return f"synapse_{secrets.token_urlsafe(16)}"
    
    @staticmethod
    def mask_secret(secret: str) -> str:
        """
        Mask webhook secret for logging/display.
        
        Args:
            secret: Secret to mask
            
        Returns:
            Masked secret string
        """
        if len(secret) <= 8:
            return "*" * len(secret)
        
        return secret[:4] + "*" * (len(secret) - 8) + secret[-4:]
    
    @staticmethod
    def validate_signature_header(signature_header: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Parse and validate signature header format.
        
        Args:
            signature_header: Signature header value
            
        Returns:
            Tuple of (is_valid, algorithm, signature)
        """
        try:
            if '=' not in signature_header:
                return False, None, None
            
            algorithm, signature = signature_header.split('=', 1)
            
            # Validate algorithm
            if algorithm not in ['sha1', 'sha256']:
                return False, None, None
            
            # Validate signature format (hex)
            if not all(c in '0123456789abcdef' for c in signature.lower()):
                return False, None, None
            
            # Validate signature length
            expected_lengths = {'sha1': 40, 'sha256': 64}
            if len(signature) != expected_lengths[algorithm]:
                return False, None, None
            
            return True, algorithm, signature
            
        except Exception:
            return False, None, None
    
    @staticmethod
    def is_safe_redirect(url: str, allowed_domains: Optional[list] = None) -> bool:
        """
        Check if a redirect URL is safe to follow.
        
        Args:
            url: URL to check
            allowed_domains: List of allowed domains for redirects
            
        Returns:
            True if redirect is safe
        """
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            
            # Must be HTTPS
            if parsed.scheme != 'https':
                return False
            
            # Check against allowed domains if provided
            if allowed_domains:
                hostname = parsed.hostname.lower() if parsed.hostname else ''
                return any(
                    hostname == domain.lower() or hostname.endswith(f'.{domain.lower()}')
                    for domain in allowed_domains
                )
            
            # Default: allow any HTTPS URL
            return True
            
        except Exception:
            return False