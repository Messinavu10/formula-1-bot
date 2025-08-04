# F1 Racing Assistant - Deployment Guide

This guide covers deploying the F1 Racing Assistant with session management and multi-user support using Docker.

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose**: Install Docker Desktop or Docker Engine with Compose
2. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### Deployment Steps

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd formula-1-bot
   ```

2. **Configure Environment**:
   ```bash
   cp env.example .env
   # Edit .env file with your OpenAI API key and other settings
   ```

3. **Deploy**:
   ```bash
   ./deploy.sh
   ```

4. **Access the Application**:
   - Web Interface: http://localhost:8080
   - API Documentation: http://localhost:8080/docs

## 📋 Environment Variables

Create a `.env` file with the following variables:

```bash
# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=f1_racing
DB_USER=f1_user
DB_PASSWORD=f1_password

# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith API Key for monitoring
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Application Settings
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

## 🏗️ Architecture

The deployment consists of:

1. **PostgreSQL Database**: Stores F1 data and session information
2. **F1 Racing Assistant**: Main application with session management
3. **Nginx** (optional): Reverse proxy for production deployments

### Session Management Features

- **Multi-User Support**: Each user gets isolated sessions
- **Automatic Cleanup**: Expired sessions are automatically removed
- **Conversation History**: Maintains context across interactions
- **Rate Limiting**: Prevents abuse with configurable limits

## 🐳 Docker Services

### PostgreSQL (`postgres`)
- **Image**: `postgres:15-alpine`
- **Port**: 5432
- **Data**: Persisted in Docker volume
- **Health Check**: Automatic readiness detection

### F1 Racing Assistant (`f1_app`)
- **Port**: 8080
- **Features**: 
  - Automatic database population
  - Session management
  - Multi-user support
  - Health monitoring

### Nginx (`nginx`) - Production Only
- **Ports**: 80, 443
- **Features**:
  - Load balancing
  - Rate limiting
  - SSL termination
  - Security headers

## 🔧 Deployment Commands

```bash
# Deploy the application
./deploy.sh

# View logs
./deploy.sh logs

# Check status
./deploy.sh status

# Stop services
./deploy.sh stop

# Restart application
./deploy.sh restart

# Clean up everything
./deploy.sh cleanup
```

## 📊 Session Management API

### Create Session
```bash
curl -X POST "http://localhost:8080/api/session/create" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

### Validate Session
```bash
curl "http://localhost:8080/api/session/{session_id}/validate"
```

### Get Session Stats
```bash
curl "http://localhost:8080/api/session/stats"
```

### Chat with Session
```bash
curl -X POST "http://localhost:8080/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me Max Verstappen performance at Monaco GP",
    "session_id": "your-session-id"
  }'
```

## 🔒 Security Features

### Rate Limiting
- **API Endpoints**: 10 requests/second
- **Chat Endpoint**: 5 requests/second
- **Burst Allowance**: 20 requests for API, 10 for chat

### Security Headers
- X-Frame-Options: SAMEORIGIN
- X-XSS-Protection: 1; mode=block
- X-Content-Type-Options: nosniff
- Content-Security-Policy: Configured for security

## 📈 Monitoring

### Health Checks
- **Application**: `GET /api/health`
- **Database**: PostgreSQL readiness check
- **Nginx**: Automatic health monitoring

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs f1_app
docker-compose logs postgres

# Follow logs in real-time
docker-compose logs -f
```

## 🚀 Production Deployment

### 1. SSL Configuration
Uncomment and configure SSL in `nginx.conf`:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    # ... rest of configuration
}
```

### 2. Environment Variables
Set production environment variables:
```bash
# Production settings
DOMAIN=your-domain.com
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Session management
MAX_SESSIONS=1000
SESSION_CLEANUP_INTERVAL=300
MAX_SESSION_AGE_HOURS=24
```

### 3. Deploy with Nginx
```bash
docker-compose --profile production up -d
```

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps postgres
   
   # Check logs
   docker-compose logs postgres
   ```

2. **Application Not Starting**
   ```bash
   # Check application logs
   docker-compose logs f1_app
   
   # Check environment variables
   docker-compose exec f1_app env | grep DB_
   ```

3. **Session Issues**
   ```bash
   # Check session stats
   curl http://localhost:8080/api/session/stats
   
   # Restart application
   ./deploy.sh restart
   ```

### Performance Tuning

1. **Database Optimization**
   ```sql
   -- Add indexes for better performance
   CREATE INDEX idx_sessions_user_id ON sessions(user_id);
   CREATE INDEX idx_conversations_session_id ON conversations(session_id);
   ```

2. **Application Scaling**
   ```yaml
   # In docker-compose.yml, add more app instances
   f1_app_2:
     build: .
     # ... same configuration as f1_app
   ```

## 📝 Development

### Local Development
```bash
# Run without Docker
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Testing
```bash
# Test session management
curl -X POST "http://localhost:8080/api/session/create"
curl -X POST "http://localhost:8080/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "test message", "session_id": "test-session"}'
```

## 🤝 Support

For issues and questions:
1. Check the logs: `./deploy.sh logs`
2. Verify environment variables
3. Check Docker container status: `docker-compose ps`
4. Review the troubleshooting section above

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 