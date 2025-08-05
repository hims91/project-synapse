"""
Secure credential management and secrets handling for Project Synapse.

Provides secure storage, retrieval, and management of sensitive data
including API keys, database credentials, encryption keys, and other secrets.
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import keyring
from pathlib import Path

from ..logging_config import get_logger


class SecretType(str, Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    OAUTH_TOKEN = "oauth_token"
    WEBHOOK_SECRET = "webhook_secret"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    GENERIC = "generic"


class SecretScope(str, Enum):
    """Scope of secret access."""
    GLOBAL = "global"
    SERVICE = "service"
    USER = "user"
    ENVIRONMENT = "environment"


@dataclass
class SecretMetadata:
    """Metadata for a secret."""
    name: str
    secret_type: SecretType
    scope: SecretScope
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    description: Optional[str] = None
    tags: Dict[str, str] = None
    rotation_interval_days: Optional[int] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    def is_expired(self) -> bool:
        """Check if secret is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def needs_rotation(self) -> bool:
        """Check if secret needs rotation."""
        if self.rotation_interval_days is None:
            return False
        
        rotation_due = self.updated_at + timedelta(days=self.rotation_interval_days)
        return datetime.utcnow() > rotation_due


@dataclass
class Secret:
    """A secret with its metadata and encrypted value."""
    metadata: SecretMetadata
    encrypted_value: bytes
    salt: bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'metadata': self.metadata.to_dict(),
            'encrypted_value': base64.b64encode(self.encrypted_value).decode('utf-8'),
            'salt': base64.b64encode(self.salt).decode('utf-8')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Secret':
        """Create Secret from dictionary."""
        metadata_data = data['metadata']
        
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'updated_at', 'expires_at', 'last_accessed']:
            if metadata_data.get(key):
                metadata_data[key] = datetime.fromisoformat(metadata_data[key])
        
        metadata = SecretMetadata(**metadata_data)
        
        return cls(
            metadata=metadata,
            encrypted_value=base64.b64decode(data['encrypted_value']),
            salt=base64.b64decode(data['salt'])
        )


class EncryptionProvider:
    """Handles encryption and decryption of secrets."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.logger = get_logger(__name__, 'encryption_provider')
        self.master_key = master_key or self._get_or_create_master_key()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        # Try to get from environment variable first
        env_key = os.getenv('SYNAPSE_MASTER_KEY')
        if env_key:
            try:
                return base64.b64decode(env_key)
            except Exception:
                self.logger.warning("Invalid master key in environment variable")
        
        # Try to get from keyring
        try:
            stored_key = keyring.get_password("synapse", "master_key")
            if stored_key:
                return base64.b64decode(stored_key)
        except Exception:
            self.logger.warning("Could not retrieve master key from keyring")
        
        # Generate new master key
        master_key = secrets.token_bytes(32)  # 256-bit key
        
        # Try to store in keyring
        try:
            keyring.set_password("synapse", "master_key", base64.b64encode(master_key).decode())
            self.logger.info("Generated and stored new master key in keyring")
        except Exception:
            self.logger.warning("Could not store master key in keyring")
        
        return master_key
    
    def encrypt(self, plaintext: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """Encrypt plaintext with optional salt."""
        if salt is None:
            salt = secrets.token_bytes(16)  # 128-bit salt
        
        # Derive key from master key and salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(self.master_key)
        
        # Encrypt using Fernet
        fernet = Fernet(base64.urlsafe_b64encode(key))
        encrypted = fernet.encrypt(plaintext.encode('utf-8'))
        
        return encrypted, salt
    
    def decrypt(self, encrypted_data: bytes, salt: bytes) -> str:
        """Decrypt encrypted data using salt."""
        # Derive key from master key and salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(self.master_key)
        
        # Decrypt using Fernet
        fernet = Fernet(base64.urlsafe_b64encode(key))
        decrypted = fernet.decrypt(encrypted_data)
        
        return decrypted.decode('utf-8')
    
    def generate_key(self, key_type: str = "generic") -> str:
        """Generate a secure random key."""
        if key_type == "api_key":
            # Generate API key format: prefix + random string
            prefix = "sk_"
            random_part = secrets.token_urlsafe(32)
            return f"{prefix}{random_part}"
        elif key_type == "jwt_secret":
            # Generate JWT secret (256-bit)
            return secrets.token_urlsafe(32)
        elif key_type == "webhook_secret":
            # Generate webhook secret
            return secrets.token_hex(32)
        elif key_type == "encryption_key":
            # Generate encryption key (256-bit)
            return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
        else:
            # Generic secure random string
            return secrets.token_urlsafe(32)


class SecretsManager:
    """Manages secure storage and retrieval of secrets."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = get_logger(__name__, 'secrets_manager')
        self.encryption_provider = EncryptionProvider()
        
        # Storage configuration
        self.storage_path = Path(storage_path or os.getenv('SYNAPSE_SECRETS_PATH', '.secrets'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed secrets
        self._cache: Dict[str, tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes
        
        # Statistics
        self.stats = {
            'secrets_stored': 0,
            'secrets_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rotations_performed': 0,
            'expired_secrets_cleaned': 0
        }
        
        # Load existing secrets metadata
        self._load_secrets_index()
    
    def _load_secrets_index(self):
        """Load secrets metadata index."""
        index_file = self.storage_path / 'index.json'
        self.secrets_index: Dict[str, SecretMetadata] = {}
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                
                for name, metadata_dict in index_data.items():
                    # Convert ISO strings back to datetime objects
                    for key in ['created_at', 'updated_at', 'expires_at', 'last_accessed']:
                        if metadata_dict.get(key):
                            metadata_dict[key] = datetime.fromisoformat(metadata_dict[key])
                    
                    self.secrets_index[name] = SecretMetadata(**metadata_dict)
                
                self.logger.info(f"Loaded {len(self.secrets_index)} secrets from index")
            
            except Exception as e:
                self.logger.error(f"Error loading secrets index: {e}")
                self.secrets_index = {}
    
    def _save_secrets_index(self):
        """Save secrets metadata index."""
        index_file = self.storage_path / 'index.json'
        
        try:
            index_data = {
                name: metadata.to_dict()
                for name, metadata in self.secrets_index.items()
            }
            
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving secrets index: {e}")
    
    def store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
        scope: SecretScope = SecretScope.GLOBAL,
        description: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        rotation_interval_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        overwrite: bool = False
    ) -> bool:
        """
        Store a secret securely.
        
        Args:
            name: Unique name for the secret
            value: Secret value to store
            secret_type: Type of secret
            scope: Access scope
            description: Optional description
            expires_at: Optional expiration date
            rotation_interval_days: Days between rotations
            tags: Optional tags for categorization
            overwrite: Whether to overwrite existing secret
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Check if secret already exists
            if name in self.secrets_index and not overwrite:
                self.logger.warning(f"Secret '{name}' already exists and overwrite=False")
                return False
            
            # Create metadata
            now = datetime.utcnow()
            metadata = SecretMetadata(
                name=name,
                secret_type=secret_type,
                scope=scope,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                description=description,
                tags=tags or {},
                rotation_interval_days=rotation_interval_days
            )
            
            # Encrypt the secret
            encrypted_value, salt = self.encryption_provider.encrypt(value)
            
            # Create secret object
            secret = Secret(
                metadata=metadata,
                encrypted_value=encrypted_value,
                salt=salt
            )
            
            # Save to file
            secret_file = self.storage_path / f"{name}.json"
            with open(secret_file, 'w') as f:
                json.dump(secret.to_dict(), f, indent=2)
            
            # Update index
            self.secrets_index[name] = metadata
            self._save_secrets_index()
            
            # Clear cache entry if exists
            if name in self._cache:
                del self._cache[name]
            
            self.stats['secrets_stored'] += 1
            
            self.logger.info(
                f"Stored secret '{name}'",
                operation="store_secret",
                secret_type=secret_type.value,
                scope=scope.value
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing secret '{name}': {e}")
            return False
    
    def get_secret(self, name: str, use_cache: bool = True) -> Optional[str]:
        """
        Retrieve a secret value.
        
        Args:
            name: Name of the secret
            use_cache: Whether to use cache
            
        Returns:
            Secret value or None if not found
        """
        try:
            # Check cache first
            if use_cache and name in self._cache:
                cached_value, cached_time = self._cache[name]
                if datetime.utcnow() - cached_time < self._cache_ttl:
                    self.stats['cache_hits'] += 1
                    self._update_access_stats(name)
                    return cached_value
                else:
                    # Cache expired
                    del self._cache[name]
            
            self.stats['cache_misses'] += 1
            
            # Check if secret exists in index
            if name not in self.secrets_index:
                self.logger.warning(f"Secret '{name}' not found in index")
                return None
            
            metadata = self.secrets_index[name]
            
            # Check if secret is expired
            if metadata.is_expired():
                self.logger.warning(f"Secret '{name}' is expired")
                return None
            
            # Load secret from file
            secret_file = self.storage_path / f"{name}.json"
            if not secret_file.exists():
                self.logger.error(f"Secret file for '{name}' not found")
                return None
            
            with open(secret_file, 'r') as f:
                secret_data = json.load(f)
            
            secret = Secret.from_dict(secret_data)
            
            # Decrypt the value
            decrypted_value = self.encryption_provider.decrypt(
                secret.encrypted_value,
                secret.salt
            )
            
            # Cache the value
            if use_cache:
                self._cache[name] = (decrypted_value, datetime.utcnow())
            
            # Update access statistics
            self._update_access_stats(name)
            
            self.stats['secrets_retrieved'] += 1
            
            return decrypted_value
        
        except Exception as e:
            self.logger.error(f"Error retrieving secret '{name}': {e}")
            return None
    
    def _update_access_stats(self, name: str):
        """Update access statistics for a secret."""
        if name in self.secrets_index:
            metadata = self.secrets_index[name]
            metadata.last_accessed = datetime.utcnow()
            metadata.access_count += 1
            self._save_secrets_index()
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Name of the secret to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Check if secret exists
            if name not in self.secrets_index:
                self.logger.warning(f"Secret '{name}' not found")
                return False
            
            # Remove from index
            del self.secrets_index[name]
            self._save_secrets_index()
            
            # Remove file
            secret_file = self.storage_path / f"{name}.json"
            if secret_file.exists():
                secret_file.unlink()
            
            # Remove from cache
            if name in self._cache:
                del self._cache[name]
            
            self.logger.info(f"Deleted secret '{name}'", operation="delete_secret")
            return True
        
        except Exception as e:
            self.logger.error(f"Error deleting secret '{name}': {e}")
            return False
    
    def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
        scope: Optional[SecretScope] = None,
        include_expired: bool = False
    ) -> List[SecretMetadata]:
        """
        List secrets with optional filtering.
        
        Args:
            secret_type: Filter by secret type
            scope: Filter by scope
            include_expired: Whether to include expired secrets
            
        Returns:
            List of secret metadata
        """
        secrets = []
        
        for metadata in self.secrets_index.values():
            # Apply filters
            if secret_type and metadata.secret_type != secret_type:
                continue
            
            if scope and metadata.scope != scope:
                continue
            
            if not include_expired and metadata.is_expired():
                continue
            
            secrets.append(metadata)
        
        return sorted(secrets, key=lambda x: x.name)
    
    def rotate_secret(self, name: str, new_value: Optional[str] = None) -> bool:
        """
        Rotate a secret (generate new value or use provided one).
        
        Args:
            name: Name of the secret to rotate
            new_value: New value (if None, will be generated)
            
        Returns:
            True if rotated successfully, False otherwise
        """
        try:
            # Check if secret exists
            if name not in self.secrets_index:
                self.logger.warning(f"Secret '{name}' not found for rotation")
                return False
            
            metadata = self.secrets_index[name]
            
            # Generate new value if not provided
            if new_value is None:
                new_value = self.encryption_provider.generate_key(metadata.secret_type.value)
            
            # Store the rotated secret
            success = self.store_secret(
                name=name,
                value=new_value,
                secret_type=metadata.secret_type,
                scope=metadata.scope,
                description=metadata.description,
                expires_at=metadata.expires_at,
                rotation_interval_days=metadata.rotation_interval_days,
                tags=metadata.tags,
                overwrite=True
            )
            
            if success:
                self.stats['rotations_performed'] += 1
                self.logger.info(f"Rotated secret '{name}'", operation="rotate_secret")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error rotating secret '{name}': {e}")
            return False
    
    def cleanup_expired_secrets(self) -> int:
        """
        Clean up expired secrets.
        
        Returns:
            Number of secrets cleaned up
        """
        expired_secrets = []
        
        for name, metadata in self.secrets_index.items():
            if metadata.is_expired():
                expired_secrets.append(name)
        
        cleaned_count = 0
        for name in expired_secrets:
            if self.delete_secret(name):
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.stats['expired_secrets_cleaned'] += cleaned_count
            self.logger.info(f"Cleaned up {cleaned_count} expired secrets")
        
        return cleaned_count
    
    def check_rotation_needed(self) -> List[str]:
        """
        Check which secrets need rotation.
        
        Returns:
            List of secret names that need rotation
        """
        rotation_needed = []
        
        for name, metadata in self.secrets_index.items():
            if metadata.needs_rotation():
                rotation_needed.append(name)
        
        return rotation_needed
    
    def export_secrets(self, include_values: bool = False) -> Dict[str, Any]:
        """
        Export secrets metadata (and optionally values).
        
        Args:
            include_values: Whether to include secret values (DANGEROUS!)
            
        Returns:
            Dictionary of secrets data
        """
        export_data = {
            'metadata': {
                name: metadata.to_dict()
                for name, metadata in self.secrets_index.items()
            },
            'exported_at': datetime.utcnow().isoformat(),
            'include_values': include_values
        }
        
        if include_values:
            self.logger.warning("Exporting secrets with values - ensure secure handling!")
            export_data['values'] = {}
            for name in self.secrets_index:
                value = self.get_secret(name, use_cache=False)
                if value:
                    export_data['values'][name] = value
        
        return export_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get secrets manager statistics."""
        return {
            **self.stats,
            'total_secrets': len(self.secrets_index),
            'cached_secrets': len(self._cache),
            'expired_secrets': len([m for m in self.secrets_index.values() if m.is_expired()]),
            'rotation_needed': len(self.check_rotation_needed())
        }


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def store_secret(name: str, value: str, secret_type: SecretType = SecretType.GENERIC, **kwargs) -> bool:
    """Store a secret using the global manager."""
    manager = get_secrets_manager()
    return manager.store_secret(name, value, secret_type, **kwargs)


def get_secret(name: str) -> Optional[str]:
    """Get a secret using the global manager."""
    manager = get_secrets_manager()
    return manager.get_secret(name)


def rotate_secret(name: str, new_value: Optional[str] = None) -> bool:
    """Rotate a secret using the global manager."""
    manager = get_secrets_manager()
    return manager.rotate_secret(name, new_value)


def generate_api_key() -> str:
    """Generate a secure API key."""
    provider = EncryptionProvider()
    return provider.generate_key("api_key")


def generate_jwt_secret() -> str:
    """Generate a secure JWT secret."""
    provider = EncryptionProvider()
    return provider.generate_key("jwt_secret")