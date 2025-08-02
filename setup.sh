#!/bin/bash

# Fake News Detection Docker Setup Script
# This script helps you set up and manage the Docker containers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi

    print_status "Docker and Docker Compose are installed ✓"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p logs data nginx/ssl
    touch logs/app.log
    print_status "Directories created ✓"
}

# Setup environment file
setup_env() {
    if [ ! -f .env ]; then
        print_status "Creating .env file from .env.example..."
        cp .env.example .env
        print_warning "Please review and update the .env file with your settings"
    else
        print_status ".env file already exists ✓"
    fi
}

# Build and start services
start_development() {
    print_header "Starting Development Environment"
    check_docker
    create_directories
    setup_env

    print_status "Building and starting development containers..."
    docker-compose -f docker-compose.dev.yml up --build -d

    print_status "Development environment started! ✓"
    echo ""
    echo "🚀 Application is running at: http://localhost:8000"
    echo "📊 Redis is running at: localhost:6379"
    echo ""
    echo "To view logs: docker-compose -f docker-compose.dev.yml logs -f"
    echo "To stop: docker-compose -f docker-compose.dev.yml down"
}

start_production() {
    print_header "Starting Production Environment"
    check_docker
    create_directories
    setup_env

    print_status "Building and starting production containers..."
    docker-compose -f docker-compose.prod.yml up --build -d

    print_status "Production environment started! ✓"
    echo ""
    echo "🚀 Application is running at: http://localhost:8000"
    echo "🌐 Nginx is running at: http://localhost:80"
    echo ""
    echo "To view logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "To stop: docker-compose -f docker-compose.prod.yml down"
}

# Stop all services
stop_services() {
    print_header "Stopping All Services"

    print_status "Stopping development environment..."
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true

    print_status "Stopping production environment..."  
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true

    print_status "All services stopped ✓"
}

# Clean up containers, images, and volumes
cleanup() {
    print_header "Cleaning Up Docker Resources"

    print_warning "This will remove all containers, images, and volumes related to this project"
    read -p "Are you sure? (y/N): " confirm

    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        print_status "Stopping all services..."
        stop_services

        print_status "Removing containers and images..."
        docker system prune -f
        docker volume prune -f

        # Remove project-specific volumes
        docker volume rm $(docker volume ls -q | grep fake) 2>/dev/null || true

        print_status "Cleanup completed ✓"
    else
        print_status "Cleanup cancelled"
    fi
}

# Show logs
show_logs() {
    local env=${1:-dev}
    if [ "$env" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml logs -f
    else
        docker-compose -f docker-compose.dev.yml logs -f
    fi
}

# Show container status
show_status() {
    print_header "Container Status"
    docker ps --filter "name=fake_news" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Install dependencies locally (for development)
install_deps() {
    print_header "Installing Local Dependencies"

    if command -v python3 &> /dev/null; then
        print_status "Installing Python dependencies..."
        pip3 install -r requirements.txt
        print_status "Dependencies installed ✓"
    else
        print_error "Python3 is not installed"
        exit 1
    fi
}

# Test API endpoints
test_api() {
    print_header "Testing API Endpoints"

    local base_url="http://localhost:8000"

    print_status "Testing health endpoint..."
    if curl -s "$base_url/health" > /dev/null; then
        print_status "Health endpoint: ✓"
    else
        print_error "Health endpoint: ✗"
    fi

    print_status "Testing prediction endpoint..."
    if curl -s -X POST -H "Content-Type: application/json" \
            -d '{"text": "This is a test news article"}' \
            "$base_url/predict" > /dev/null; then
        print_status "Prediction endpoint: ✓"
    else
        print_error "Prediction endpoint: ✗"
    fi
}

# Show help
show_help() {
    echo "Fake News Detection Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start-dev     Start development environment"
    echo "  start-prod    Start production environment"
    echo "  stop          Stop all services"
    echo "  restart-dev   Restart development environment"
    echo "  restart-prod  Restart production environment"
    echo "  logs [env]    Show logs (dev/prod, default: dev)"
    echo "  status        Show container status"
    echo "  test          Test API endpoints"
    echo "  install       Install local Python dependencies"
    echo "  cleanup       Remove all containers and volumes"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start-dev"
    echo "  $0 logs prod"
    echo "  $0 status"
}

# Main script logic
case "${1:-}" in
    "start-dev")
        start_development
        ;;
    "start-prod")
        start_production
        ;;
    "stop")
        stop_services
        ;;
    "restart-dev")
        stop_services
        start_development
        ;;
    "restart-prod")
        stop_services
        start_production
        ;;
    "logs")
        show_logs $2
        ;;
    "status")
        show_status
        ;;
    "test")
        test_api
        ;;
    "install")
        install_deps
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
