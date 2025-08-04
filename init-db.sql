-- Initialize F1 Racing Database
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Set timezone
SET timezone = 'UTC';

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE f1_racing TO f1_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO f1_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO f1_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO f1_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO f1_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO f1_user;

-- Create indexes for better performance
-- (These will be created by the application when tables are created)

-- Log the initialization
DO $$
BEGIN
    RAISE NOTICE 'F1 Racing Database initialized successfully';
END $$; 