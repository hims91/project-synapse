"""
Minimal Configuration for Central Hub Deployment
Simplified config that avoids Pydantic validation issues.
"""
import os
from typing import Optional


class SimpleSettings:
    """Simple settings class without Pydantic validation."""
    
    def __init__(self):
        # Application settings
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.app_name = os.getenv("APP_NAME", "Project Synapse - Central Hub")
        self.app_version = os.getenv("APP_VERSION", "2.2.0")
        
        # Database settings
        self.database_url = os.getenv("DATABASE_URL")
        
        # Redis settings
        self.redis_url = os.getenv("REDIS_URL")
        
        # Security settings
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", self.secret_key)
        
        # API settings
        self.api_version = os.getenv("API_VERSION", "v1")
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
    
    def is_development(self) -> bool:
        return self.environment == "development"
    
    def is_production(self) -> bool:
        return self.environment == "production"


# Global settings instance
settings = SimpleSettings()


def get_settings() -> SimpleSettings:
    """Get application settings."""
    return settings