-- Price Stradamus Database Initialization
-- This script runs automatically when the PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS ml_data;

-- Set search path
SET search_path TO market_data, ml_data, public;

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA market_data TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA ml_data TO postgres;

-- Comment on schemas
COMMENT ON SCHEMA market_data IS 'Raw and processed market data from exchanges';
COMMENT ON SCHEMA ml_data IS 'Machine learning models, predictions, and evaluations';
