#!/bin/bash

# F1 Racing Assistant Deployment Script
# This script handles the complete deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Function to check environment variables
check_env() {
    print_status "Checking environment variables..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f env.example ]; then
            cp env.example .env
            print_warning "Please edit .env file with your actual values before continuing"
            print_warning "Especially set your OPENAI_API_KEY"
            exit 1
        else
            print_error "env.example not found. Please create a .env file manually"
            exit 1
        fi
    fi
    
    # Check for required environment variables
    source .env
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        print_error "OPENAI_API_KEY is not set in .env file"
        exit 1
    fi
    
    print_success "Environment variables are properly configured"
}

# Function to build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    # Stop any existing containers
    docker-compose down --remove-orphans
    
    # Build the application
    print_status "Building F1 Racing Assistant..."
    docker-compose build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose up -d
    
    print_success "Services started successfully"
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    timeout=60
    counter=0
    while ! docker-compose exec -T postgres pg_isready -U $DB_USER -d $DB_NAME > /dev/null 2>&1; do
        sleep 2
        counter=$((counter + 2))
        if [ $counter -ge $timeout ]; then
            print_error "PostgreSQL failed to start within $timeout seconds"
            exit 1
        fi
    done
    print_success "PostgreSQL is ready"
    
    # Wait for application
    print_status "Waiting for F1 Racing Assistant..."
    timeout=120
    counter=0
    while ! curl -f http://localhost:8080/api/health > /dev/null 2>&1; do
        sleep 5
        counter=$((counter + 5))
        if [ $counter -ge $timeout ]; then
            print_error "F1 Racing Assistant failed to start within $timeout seconds"
            print_status "Checking logs..."
            docker-compose logs f1_app
            exit 1
        fi
    done
    print_success "F1 Racing Assistant is ready"
}

# Function to show deployment status
show_status() {
    print_status "Deployment Status:"
    echo "=================="
    
    # Show running containers
    print_status "Running containers:"
    docker-compose ps
    
    # Show application health
    print_status "Application health:"
    if curl -f http://localhost:8080/api/health > /dev/null 2>&1; then
        print_success "Application is healthy"
    else
        print_error "Application is not responding"
    fi
    
    # Show session stats
    print_status "Session statistics:"
    curl -s http://localhost:8080/api/session/stats | python -m json.tool 2>/dev/null || print_warning "Could not fetch session stats"
    
    echo ""
    print_success "F1 Racing Assistant is deployed and running!"
    print_status "Access the application at: http://localhost:8080"
    print_status "API documentation at: http://localhost:8080/docs"
}

# Function to show logs
show_logs() {
    print_status "Showing recent logs..."
    docker-compose logs --tail=50 -f
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Main script
main() {
    print_status "Starting F1 Racing Assistant deployment..."
    
    case "${1:-deploy}" in
        "deploy")
            check_docker
            check_env
            deploy_services
            wait_for_services
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_services
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            show_status
            ;;
        "restart")
            stop_services
            deploy_services
            wait_for_services
            show_status
            ;;
        *)
            echo "Usage: $0 {deploy|logs|stop|cleanup|status|restart}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the application (default)"
            echo "  logs     - Show application logs"
            echo "  stop     - Stop all services"
            echo "  cleanup  - Stop and remove all containers and volumes"
            echo "  status   - Show deployment status"
            echo "  restart  - Restart the application"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 