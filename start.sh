#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting F1 Racing Assistant..."

# Function to wait for PostgreSQL to be ready
wait_for_postgres() {
    echo "⏳ Waiting for PostgreSQL to be ready..."
    until python -c "
import os
import psycopg2
import time

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.close()
    print('PostgreSQL is ready!')
    exit(0)
except Exception as e:
    print(f'PostgreSQL is unavailable: {e}')
    exit(1)
"; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 2
    done
    echo "✅ PostgreSQL is ready!"
}

# Function to update database with latest F1 data
update_database() {
    echo "📊 Updating database with latest F1 data from API..."
    python main.py
    if [ $? -eq 0 ]; then
        echo "✅ Database update completed successfully!"
    else
        echo "❌ Database update failed!"
        exit 1
    fi
}

# Function to start the application
start_application() {
    echo "🌐 Starting F1 Racing Assistant web application..."
    port=${PORT:-8080}
    python -m uvicorn app:app --host 0.0.0.0 --port $port
}

# Main execution
echo "🔧 Environment check..."
echo "DB_HOST: $DB_HOST"
echo "DB_PORT: $DB_PORT"
echo "DB_NAME: $DB_NAME"
echo "DB_USER: $DB_USER"

# Wait for PostgreSQL
wait_for_postgres

# Always update database with latest data
echo "🔄 Updating database with latest F1 data..."
update_database

# Start the application
start_application 