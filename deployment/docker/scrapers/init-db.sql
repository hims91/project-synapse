-- Project Synapse Database Initialization Script
-- Creates necessary extensions and initial configuration

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- Create full-text search configuration for better search
CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS english_unaccent (COPY = english);
ALTER TEXT SEARCH CONFIGURATION english_unaccent
    ALTER MAPPING FOR hword, hword_part, word
    WITH unaccent, english_stem;

-- Create indexes for better performance (will be created by Alembic migrations)
-- This script just ensures the database is ready for the application

-- Set default timezone
SET timezone = 'UTC';

-- Create application user if not exists (for production use)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'synapse_app') THEN
        CREATE ROLE synapse_app WITH LOGIN PASSWORD 'synapse_app_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE synapse TO synapse_app;
GRANT USAGE ON SCHEMA public TO synapse_app;
GRANT CREATE ON SCHEMA public TO synapse_app;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Log initialization
INSERT INTO pg_stat_statements_reset() VALUES (DEFAULT) ON CONFLICT DO NOTHING;

-- Vacuum and analyze for optimal performance
VACUUM ANALYZE;