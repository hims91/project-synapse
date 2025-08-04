"""
Synaptic Vesicle - Database Models
Layer 2: Signal Network

This module contains all database models for Project Synapse.
The Synaptic Vesicle serves as the PostgreSQL queue, recipe memory, and feed list.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Text, Integer, DateTime, Boolean, DECIMAL, 
    JSON, ARRAY, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Article(Base):
    """
    Articles table with full-text search capabilities.
    Stores scraped web content with NLP enrichment and metadata.
    """
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    summary = Column(Text)
    author = Column(String(255))
    published_at = Column(DateTime(timezone=True))
    scraped_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    source_domain = Column(String(255), nullable=False, index=True)
    
    # NLP enrichment data stored as JSON
    nlp_data = Column(JSON, default=dict)
    
    # Technical metadata about the webpage
    page_metadata = Column(JSON, default=dict)
    
    # Full-text search vector (generated automatically)
    search_vector = Column(TSVECTOR)

    # Indexes for performance
    __table_args__ = (
        Index('ix_articles_published_at', 'published_at'),
        Index('ix_articles_scraped_at', 'scraped_at'),
        Index('ix_articles_source_domain', 'source_domain'),
        Index('ix_articles_search_vector', 'search_vector', postgresql_using='gin'),
        Index('ix_articles_nlp_sentiment', 'nlp_data', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:50]}...', domain='{self.source_domain}')>"


class ScrapingRecipe(Base):
    """
    Scraping recipes with success tracking.
    Stores CSS selectors and actions for extracting content from specific domains.
    """
    __tablename__ = "scraping_recipes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain = Column(String(255), unique=True, nullable=False, index=True)
    
    # CSS selectors for content extraction
    selectors = Column(JSON, nullable=False)
    
    # Optional actions for dynamic content (JavaScript execution, etc.)
    actions = Column(JSON, default=list)
    
    # Success tracking
    success_rate = Column(DECIMAL(3, 2), default=0.0)
    usage_count = Column(Integer, default=0)
    
    # Metadata
    last_updated = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    created_by = Column(String(50), default='learning')  # 'learning' or 'manual'

    # Constraints
    __table_args__ = (
        CheckConstraint('success_rate >= 0.0 AND success_rate <= 1.0', name='valid_success_rate'),
        CheckConstraint('usage_count >= 0', name='non_negative_usage_count'),
        Index('ix_scraping_recipes_success_rate', 'success_rate'),
        Index('ix_scraping_recipes_last_updated', 'last_updated'),
    )

    def __repr__(self):
        return f"<ScrapingRecipe(domain='{self.domain}', success_rate={self.success_rate}, usage_count={self.usage_count})>"


class TaskQueue(Base):
    """
    Task queue with priority and retry logic.
    Manages asynchronous tasks for scraping, processing, and analysis.
    """
    __tablename__ = "task_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type = Column(String(100), nullable=False, index=True)
    
    # Task payload (URL, parameters, etc.)
    payload = Column(JSON, nullable=False)
    
    # Priority system (1 = highest, 10 = lowest)
    priority = Column(Integer, default=5, index=True)
    
    # Status tracking
    status = Column(String(50), default='pending', nullable=False, index=True)
    
    # Retry logic
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timing
    scheduled_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True))
    
    # Error tracking
    error_message = Column(Text)

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('priority >= 1 AND priority <= 10', name='valid_priority'),
        CheckConstraint('retry_count >= 0', name='non_negative_retry_count'),
        CheckConstraint('max_retries >= 0', name='non_negative_max_retries'),
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", name='valid_status'),
        Index('ix_task_queue_status_priority', 'status', 'priority'),
        Index('ix_task_queue_scheduled_at', 'scheduled_at'),
        Index('ix_task_queue_task_type', 'task_type'),
    )

    def __repr__(self):
        return f"<TaskQueue(id={self.id}, type='{self.task_type}', status='{self.status}', priority={self.priority})>"


class MonitoringSubscription(Base):
    """
    Monitoring subscriptions for keyword alerts.
    Stores webhook configurations for real-time notifications.
    """
    __tablename__ = "monitoring_subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    
    # Keywords to monitor
    keywords = Column(ARRAY(String), nullable=False)
    
    # Webhook configuration
    webhook_url = Column(Text, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    last_triggered = Column(DateTime(timezone=True))

    # Indexes
    __table_args__ = (
        Index('ix_monitoring_subscriptions_user_id', 'user_id'),
        Index('ix_monitoring_subscriptions_active', 'is_active'),
        Index('ix_monitoring_subscriptions_keywords', 'keywords', postgresql_using='gin'),
        Index('ix_monitoring_subscriptions_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<MonitoringSubscription(id={self.id}, name='{self.name}', active={self.is_active})>"


class APIUsage(Base):
    """
    API usage tracking for rate limiting and analytics.
    Records all API calls with performance metrics.
    """
    __tablename__ = "api_usage"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Request details
    endpoint = Column(String(255), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    
    # Response details
    status_code = Column(Integer, index=True)
    response_time_ms = Column(Integer)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)

    # Indexes for analytics and rate limiting
    __table_args__ = (
        Index('ix_api_usage_api_key_timestamp', 'api_key_id', 'timestamp'),
        Index('ix_api_usage_endpoint_timestamp', 'endpoint', 'timestamp'),
        Index('ix_api_usage_status_code', 'status_code'),
    )

    def __repr__(self):
        return f"<APIUsage(endpoint='{self.endpoint}', method='{self.method}', status={self.status_code})>"


class Feed(Base):
    """
    RSS/Atom feeds configuration for the Dendrites layer.
    Stores feed URLs with polling configuration and metadata.
    """
    __tablename__ = "feeds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    
    # Feed categorization
    category = Column(String(100), index=True)
    
    # Polling configuration
    priority = Column(Integer, default=5, index=True)  # 1 = high priority (60s), 10 = low priority (10min)
    polling_interval = Column(Integer, default=300)  # seconds
    
    # Status tracking
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_polled = Column(DateTime(timezone=True))
    last_successful_poll = Column(DateTime(timezone=True))
    consecutive_failures = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('priority >= 1 AND priority <= 10', name='valid_feed_priority'),
        CheckConstraint('polling_interval > 0', name='positive_polling_interval'),
        CheckConstraint('consecutive_failures >= 0', name='non_negative_failures'),
        Index('ix_feeds_priority_active', 'priority', 'is_active'),
        Index('ix_feeds_category', 'category'),
        Index('ix_feeds_last_polled', 'last_polled'),
    )

    def __repr__(self):
        return f"<Feed(name='{self.name}', priority={self.priority}, active={self.is_active})>"


class User(Base):
    """
    User accounts for API access and dashboard authentication.
    Stores user information and API tier configuration.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    
    # API configuration
    api_key = Column(String(255), unique=True, nullable=False, index=True)
    tier = Column(String(50), default='free', nullable=False)  # 'free', 'premium', 'enterprise'
    
    # Usage limits
    monthly_api_calls = Column(Integer, default=0)
    api_call_limit = Column(Integer, default=1000)  # per month
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True))

    # Constraints
    __table_args__ = (
        CheckConstraint("tier IN ('free', 'premium', 'enterprise')", name='valid_tier'),
        CheckConstraint('monthly_api_calls >= 0', name='non_negative_api_calls'),
        CheckConstraint('api_call_limit > 0', name='positive_api_limit'),
        Index('ix_users_tier', 'tier'),
        Index('ix_users_is_active', 'is_active'),
    )

    def __repr__(self):
        return f"<User(username='{self.username}', tier='{self.tier}', active={self.is_active})>"


class TrendsSummary(Base):
    """
    Pre-calculated trends data for fast API responses.
    Stores trending topics and entities with velocity metrics.
    """
    __tablename__ = "trends_summary"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time window for the trend calculation
    time_window = Column(String(10), nullable=False, index=True)  # '1h', '6h', '24h'
    category = Column(String(100), index=True)
    
    # Trend data
    trending_topics = Column(JSON, nullable=False)  # [{topic, velocity, volume}, ...]
    trending_entities = Column(JSON, nullable=False)  # [{entity, type, velocity, volume}, ...]
    
    # Metadata
    generated_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Ensure we only keep recent trend data
    __table_args__ = (
        Index('ix_trends_summary_window_category', 'time_window', 'category'),
        Index('ix_trends_summary_generated_at', 'generated_at'),
    )

    def __repr__(self):
        return f"<TrendsSummary(window='{self.time_window}', category='{self.category}', generated_at={self.generated_at})>"