"""
Shared Configuration - Application Settings and Environment Management
Centralized configuration management for Project Synapse.

This module provides:
- Environment-based configuration
- Type-safe settings with validation
- Database connection settings
- API configuration
- External service configuration
"""
import os
from typing import Optional, List
from pydantic import Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings
from enum import Enum


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Primary database URL (takes precedence if set)
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    # Individual database components (used if DATABASE_URL not set)
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("synapse", env="DB_NAME")
    db_user: str = Field("synapse", env="DB_USER")
    db_password: str = Field("synapse_dev_password", env="DB_PASSWORD")
    
    # Connection pool settings
    db_pool_size: int = Field(20, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(30, env="DB_MAX_OVERFLOW")
    db_echo: bool = Field(False, env="DB_ECHO")
    
    # Connection retry settings
    db_retry_attempts: int = Field(3, env="DB_RETRY_ATTEMPTS")
    db_retry_delay: float = Field(1.0, env="DB_RETRY_DELAY")
    
    @field_validator("db_port")
    @classmethod
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator("db_pool_size")
    @classmethod
    def validate_pool_size(cls, v):
        if v < 1:
            raise ValueError("Pool size must be at least 1")
        return v
    
    def get_database_url(self) -> str:
        """Get the complete database URL."""
        if self.database_url:
            return self.database_url
        
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_pool_size: int = Field(10, env="REDIS_POOL_SIZE")
    redis_timeout: int = Field(5, env="REDIS_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class CloudflareSettings(BaseSettings):
    """Cloudflare configuration settings."""
    
    cloudflare_api_token: Optional[str] = Field(None, env="CLOUDFLARE_API_TOKEN")
    cloudflare_account_id: Optional[str] = Field(None, env="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_r2_bucket: str = Field("synapse-spinal-cord", env="CLOUDFLARE_R2_BUCKET")
    cloudflare_workers_subdomain: Optional[str] = Field(None, env="CLOUDFLARE_WORKERS_SUBDOMAIN")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SupabaseSettings(BaseSettings):
    """Supabase configuration settings (alternative to self-hosted PostgreSQL)."""
    
    supabase_url: Optional[HttpUrl] = Field(None, env="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(None, env="SUPABASE_ANON_KEY")
    supabase_service_role_key: Optional[str] = Field(None, env="SUPABASE_SERVICE_ROLE_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ExternalServiceSettings(BaseSettings):
    """External service configuration settings."""
    
    # Deployment services
    render_api_key: Optional[str] = Field(None, env="RENDER_API_KEY")
    railway_token: Optional[str] = Field(None, env="RAILWAY_TOKEN")
    vercel_token: Optional[str] = Field(None, env="VERCEL_TOKEN")
    
    # GitHub Actions
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN")
    github_repo: Optional[str] = Field(None, env="GITHUB_REPO")
    
    # Proxy services (Chameleon Network)
    proxy_providers: List[str] = Field(["brightdata", "smartproxy", "oxylabs"], env="PROXY_PROVIDERS")
    brightdata_username: Optional[str] = Field(None, env="BRIGHTDATA_USERNAME")
    brightdata_password: Optional[str] = Field(None, env="BRIGHTDATA_PASSWORD")
    
    # Tor configuration
    tor_enabled: bool = Field(False, env="TOR_ENABLED")
    tor_socks_port: int = Field(9050, env="TOR_SOCKS_PORT")
    
    # NLP services
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    spacy_model: str = Field("en_core_web_sm", env="SPACY_MODEL")
    
    @field_validator("proxy_providers", mode="before")
    @classmethod
    def parse_proxy_providers(cls, v):
        if isinstance(v, str):
            return [provider.strip() for provider in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class APISettings(BaseSettings):
    """API configuration settings."""
    
    api_version: str = Field("v1", env="API_VERSION")
    api_title: str = Field("Project Synapse API", env="API_TITLE")
    api_description: str = Field("The Definitive Web Intelligence Network", env="API_DESCRIPTION")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    free_tier_limit: int = Field(1000, env="FREE_TIER_LIMIT")
    premium_tier_limit: int = Field(10000, env="PREMIUM_TIER_LIMIT")
    enterprise_tier_limit: int = Field(100000, env="ENTERPRISE_TIER_LIMIT")
    
    # CORS settings
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(["GET", "POST", "PUT", "DELETE"], env="CORS_METHODS")
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("cors_methods", mode="before")
    @classmethod
    def parse_cors_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration settings."""
    
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    
    # Health check settings
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(10, env="HEALTH_CHECK_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field("dev-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Password requirements
    min_password_length: int = Field(8, env="MIN_PASSWORD_LENGTH")
    require_special_chars: bool = Field(True, env="REQUIRE_SPECIAL_CHARS")
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class FrontendSettings(BaseSettings):
    """Frontend configuration settings."""
    
    next_public_api_url: str = Field("http://localhost:8000", env="NEXT_PUBLIC_API_URL")
    next_public_ws_url: str = Field("ws://localhost:8000/ws", env="NEXT_PUBLIC_WS_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application settings
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    app_name: str = Field("Project Synapse", env="APP_NAME")
    app_version: str = Field("2.2.0", env="APP_VERSION")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    cloudflare: CloudflareSettings = CloudflareSettings()
    supabase: SupabaseSettings = SupabaseSettings()
    external_services: ExternalServiceSettings = ExternalServiceSettings()
    api: APISettings = APISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    frontend: FrontendSettings = FrontendSettings()
    
    @field_validator("debug")
    @classmethod
    def validate_debug_in_production(cls, v, info):
        if hasattr(info, 'data') and info.data.get("environment") == Environment.PRODUCTION and v:
            raise ValueError("Debug mode should not be enabled in production")
        return v
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow extra fields for flexibility
        extra = "allow"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings.
    This function can be used as a FastAPI dependency.
    """
    return settings


def get_database_settings() -> DatabaseSettings:
    """Get database settings."""
    return settings.database


def get_redis_settings() -> RedisSettings:
    """Get Redis settings."""
    return settings.redis


def get_api_settings() -> APISettings:
    """Get API settings."""
    return settings.api


def get_monitoring_settings() -> MonitoringSettings:
    """Get monitoring settings."""
    return settings.monitoring


def get_security_settings() -> SecuritySettings:
    """Get security settings."""
    return settings.security


# Environment-specific configuration loading
def load_config_for_environment(env: Environment) -> Settings:
    """
    Load configuration for a specific environment.
    
    Args:
        env: The environment to load configuration for
        
    Returns:
        Settings instance configured for the environment
    """
    env_file_map = {
        Environment.DEVELOPMENT: ".env",
        Environment.TESTING: ".env.test",
        Environment.STAGING: ".env.staging",
        Environment.PRODUCTION: ".env.production"
    }
    
    env_file = env_file_map.get(env, ".env")
    
    # Override the default env_file for all settings classes
    class EnvironmentSettings(Settings):
        class Config:
            env_file = env_file
            case_sensitive = False
    
    return EnvironmentSettings()


# Configuration validation
def validate_configuration() -> List[str]:
    """
    Validate the current configuration and return any errors.
    
    Returns:
        List of validation error messages
    """
    errors = []
    
    try:
        # Test database configuration
        db_url = settings.database.get_database_url()
        if not db_url:
            errors.append("Database URL is not configured")
        
        # Test required security settings
        if not settings.security.secret_key:
            errors.append("SECRET_KEY is required")
        
        # Test production-specific requirements
        if settings.is_production():
            if settings.debug:
                errors.append("Debug mode should be disabled in production")
            
            if not settings.monitoring.sentry_dsn:
                errors.append("Sentry DSN should be configured in production")
        
        # Test external service configurations
        if settings.cloudflare.cloudflare_api_token and not settings.cloudflare.cloudflare_account_id:
            errors.append("Cloudflare account ID is required when API token is provided")
        
    except Exception as e:
        errors.append(f"Configuration validation error: {str(e)}")
    
    return errors


# Configuration summary for debugging
def get_config_summary() -> dict:
    """
    Get a summary of the current configuration (without sensitive data).
    
    Returns:
        Dictionary with configuration summary
    """
    return {
        "environment": settings.environment,
        "debug": settings.debug,
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "database": {
            "host": settings.database.db_host,
            "port": settings.database.db_port,
            "name": settings.database.db_name,
            "pool_size": settings.database.db_pool_size,
        },
        "api": {
            "version": settings.api.api_version,
            "rate_limit_enabled": settings.api.rate_limit_enabled,
            "free_tier_limit": settings.api.free_tier_limit,
        },
        "monitoring": {
            "log_level": settings.monitoring.log_level,
            "prometheus_enabled": settings.monitoring.prometheus_enabled,
        },
        "external_services": {
            "cloudflare_configured": bool(settings.cloudflare.cloudflare_api_token),
            "supabase_configured": bool(settings.supabase.supabase_url),
            "proxy_providers": settings.external_services.proxy_providers,
        }
    }