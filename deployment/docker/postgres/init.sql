-- Project Synapse - PostgreSQL Initialization Script
-- This script sets up the database with required extensions and initial configuration

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create full-text search configuration for better search performance
CREATE TEXT SEARCH CONFIGURATION synapse_search (COPY = english);

-- Create function to automatically update search vectors
CREATE OR REPLACE FUNCTION update_article_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('synapse_search', 
        coalesce(NEW.title, '') || ' ' || coalesce(NEW.content, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create function for trend calculation (will be used by background jobs)
CREATE OR REPLACE FUNCTION calculate_trend_velocity(
    topic_name TEXT,
    current_window INTERVAL,
    previous_window INTERVAL
) RETURNS DECIMAL AS $$
DECLARE
    current_count INTEGER;
    previous_count INTEGER;
    velocity DECIMAL;
BEGIN
    -- Count mentions in current window
    SELECT COUNT(*) INTO current_count
    FROM articles
    WHERE scraped_at >= NOW() - current_window
    AND (title ILIKE '%' || topic_name || '%' OR content ILIKE '%' || topic_name || '%');
    
    -- Count mentions in previous window
    SELECT COUNT(*) INTO previous_count
    FROM articles
    WHERE scraped_at >= NOW() - (current_window + previous_window)
    AND scraped_at < NOW() - current_window
    AND (title ILIKE '%' || topic_name || '%' OR content ILIKE '%' || topic_name || '%');
    
    -- Calculate velocity (rate of change)
    IF previous_count = 0 THEN
        velocity := CASE WHEN current_count > 0 THEN 10.0 ELSE 0.0 END;
    ELSE
        velocity := (current_count::DECIMAL - previous_count::DECIMAL) / previous_count::DECIMAL * 10.0;
    END IF;
    
    RETURN GREATEST(0.0, LEAST(10.0, velocity));
END;
$$ LANGUAGE plpgsql;

-- Create indexes for common query patterns (will be applied after table creation)
-- These will be created by Alembic migrations, but we define them here for reference

-- Performance optimization settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET pg_stat_statements.track = 'all';

-- Reload configuration
SELECT pg_reload_conf();