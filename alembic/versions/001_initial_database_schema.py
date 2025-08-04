"""Initial database schema with all core tables

Revision ID: 001
Revises: 
Create Date: 2025-08-04 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create articles table
    op.create_table('articles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('author', sa.String(length=255), nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scraped_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source_domain', sa.String(length=255), nullable=False),
        sa.Column('nlp_data', sa.JSON(), nullable=True),
        sa.Column('page_metadata', sa.JSON(), nullable=True),
        sa.Column('search_vector', postgresql.TSVECTOR(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_articles_published_at', 'articles', ['published_at'], unique=False)
    op.create_index('ix_articles_scraped_at', 'articles', ['scraped_at'], unique=False)
    op.create_index('ix_articles_source_domain', 'articles', ['source_domain'], unique=False)
    op.create_index('ix_articles_url', 'articles', ['url'], unique=True)
    op.create_index('ix_articles_search_vector', 'articles', ['search_vector'], unique=False, postgresql_using='gin')
    op.create_index('ix_articles_nlp_sentiment', 'articles', ['nlp_data'], unique=False, postgresql_using='gin')

    # Create scraping_recipes table
    op.create_table('scraping_recipes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=False),
        sa.Column('selectors', sa.JSON(), nullable=False),
        sa.Column('actions', sa.JSON(), nullable=True),
        sa.Column('success_rate', sa.DECIMAL(precision=3, scale=2), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_by', sa.String(length=50), nullable=True),
        sa.CheckConstraint('success_rate >= 0.0 AND success_rate <= 1.0', name='valid_success_rate'),
        sa.CheckConstraint('usage_count >= 0', name='non_negative_usage_count'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('domain')
    )
    op.create_index('ix_scraping_recipes_domain', 'scraping_recipes', ['domain'], unique=True)
    op.create_index('ix_scraping_recipes_success_rate', 'scraping_recipes', ['success_rate'], unique=False)
    op.create_index('ix_scraping_recipes_last_updated', 'scraping_recipes', ['last_updated'], unique=False)

    # Create task_queue table
    op.create_table('task_queue',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_type', sa.String(length=100), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('max_retries', sa.Integer(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.CheckConstraint('priority >= 1 AND priority <= 10', name='valid_priority'),
        sa.CheckConstraint('retry_count >= 0', name='non_negative_retry_count'),
        sa.CheckConstraint('max_retries >= 0', name='non_negative_max_retries'),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", name='valid_status'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_task_queue_status_priority', 'task_queue', ['status', 'priority'], unique=False)
    op.create_index('ix_task_queue_scheduled_at', 'task_queue', ['scheduled_at'], unique=False)
    op.create_index('ix_task_queue_task_type', 'task_queue', ['task_type'], unique=False)

    # Create monitoring_subscriptions table
    op.create_table('monitoring_subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('keywords', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('webhook_url', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_triggered', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_monitoring_subscriptions_user_id', 'monitoring_subscriptions', ['user_id'], unique=False)
    op.create_index('ix_monitoring_subscriptions_active', 'monitoring_subscriptions', ['is_active'], unique=False)
    op.create_index('ix_monitoring_subscriptions_keywords', 'monitoring_subscriptions', ['keywords'], unique=False, postgresql_using='gin')
    op.create_index('ix_monitoring_subscriptions_created_at', 'monitoring_subscriptions', ['created_at'], unique=False)

    # Create api_usage table
    op.create_table('api_usage',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('endpoint', sa.String(length=255), nullable=False),
        sa.Column('method', sa.String(length=10), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_api_usage_api_key_id', 'api_usage', ['api_key_id'], unique=False)
    op.create_index('ix_api_usage_endpoint', 'api_usage', ['endpoint'], unique=False)
    op.create_index('ix_api_usage_timestamp', 'api_usage', ['timestamp'], unique=False)
    op.create_index('ix_api_usage_api_key_timestamp', 'api_usage', ['api_key_id', 'timestamp'], unique=False)
    op.create_index('ix_api_usage_endpoint_timestamp', 'api_usage', ['endpoint', 'timestamp'], unique=False)
    op.create_index('ix_api_usage_status_code', 'api_usage', ['status_code'], unique=False)

    # Create feeds table
    op.create_table('feeds',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('polling_interval', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('last_polled', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_successful_poll', sa.DateTime(timezone=True), nullable=True),
        sa.Column('consecutive_failures', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('priority >= 1 AND priority <= 10', name='valid_feed_priority'),
        sa.CheckConstraint('polling_interval > 0', name='positive_polling_interval'),
        sa.CheckConstraint('consecutive_failures >= 0', name='non_negative_failures'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('url')
    )
    op.create_index('ix_feeds_url', 'feeds', ['url'], unique=True)
    op.create_index('ix_feeds_priority', 'feeds', ['priority'], unique=False)
    op.create_index('ix_feeds_is_active', 'feeds', ['is_active'], unique=False)
    op.create_index('ix_feeds_priority_active', 'feeds', ['priority', 'is_active'], unique=False)
    op.create_index('ix_feeds_category', 'feeds', ['category'], unique=False)
    op.create_index('ix_feeds_last_polled', 'feeds', ['last_polled'], unique=False)

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('api_key', sa.String(length=255), nullable=False),
        sa.Column('tier', sa.String(length=50), nullable=False),
        sa.Column('monthly_api_calls', sa.Integer(), nullable=True),
        sa.Column('api_call_limit', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("tier IN ('free', 'premium', 'enterprise')", name='valid_tier'),
        sa.CheckConstraint('monthly_api_calls >= 0', name='non_negative_api_calls'),
        sa.CheckConstraint('api_call_limit > 0', name='positive_api_limit'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_api_key', 'users', ['api_key'], unique=True)
    op.create_index('ix_users_tier', 'users', ['tier'], unique=False)
    op.create_index('ix_users_is_active', 'users', ['is_active'], unique=False)

    # Create trends_summary table
    op.create_table('trends_summary',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('time_window', sa.String(length=10), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('trending_topics', sa.JSON(), nullable=False),
        sa.Column('trending_entities', sa.JSON(), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_trends_summary_time_window', 'trends_summary', ['time_window'], unique=False)
    op.create_index('ix_trends_summary_window_category', 'trends_summary', ['time_window', 'category'], unique=False)
    op.create_index('ix_trends_summary_generated_at', 'trends_summary', ['generated_at'], unique=False)

    # Create trigger for automatic search vector updates
    op.execute("""
        CREATE TRIGGER update_article_search_vector_trigger
        BEFORE INSERT OR UPDATE ON articles
        FOR EACH ROW EXECUTE FUNCTION update_article_search_vector();
    """)

    # Create trigger for updated_at column on feeds
    op.execute("""
        CREATE TRIGGER update_feeds_updated_at_trigger
        BEFORE UPDATE ON feeds
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_article_search_vector_trigger ON articles;")
    op.execute("DROP TRIGGER IF EXISTS update_feeds_updated_at_trigger ON feeds;")
    
    # Drop tables in reverse order
    op.drop_table('trends_summary')
    op.drop_table('users')
    op.drop_table('feeds')
    op.drop_table('api_usage')
    op.drop_table('monitoring_subscriptions')
    op.drop_table('task_queue')
    op.drop_table('scraping_recipes')
    op.drop_table('articles')