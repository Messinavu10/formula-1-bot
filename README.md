# 🏎️ Formula 1 Racing Assistant

An AI-powered Formula 1 racing assistant with advanced session management and multi-user support.

## ✨ Features

- **🤖 AI-Powered Analysis**: Advanced F1 data analysis using OpenAI's GPT models
- **👥 Multi-User Support**: Isolated sessions for multiple users
- **💬 Conversation Memory**: Maintains context across interactions
- **📊 Real-time Visualizations**: Interactive charts and graphs
- **🔒 Session Management**: Automatic cleanup and rate limiting
- **🐳 Docker Deployment**: Easy deployment with PostgreSQL database
- **📈 Performance Monitoring**: Health checks and logging

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key

### Deploy
```bash
# Clone the repository
git clone <repository-url>
cd formula-1-bot

# Configure environment
cp env.example .env
# Edit .env with your OpenAI API key

# Deploy
./deploy.sh
```

### Access
- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

## 📋 What's New

### Session Management
- **Multi-User Isolation**: Each user gets their own session
- **Automatic Cleanup**: Expired sessions are automatically removed
- **Conversation History**: Maintains context across interactions
- **Rate Limiting**: Prevents abuse with configurable limits

### Deployment Improvements
- **PostgreSQL Database**: Containerized database with automatic initialization
- **Health Checks**: Automatic monitoring of all services
- **Production Ready**: Nginx reverse proxy with SSL support
- **Easy Deployment**: One-command deployment script

## 🔧 API Endpoints

### Session Management
- `POST /api/session/create` - Create new session
- `GET /api/session/{id}/validate` - Validate session
- `DELETE /api/session/{id}` - End session
- `GET /api/session/stats` - Get session statistics

### Chat Interface
- `POST /api/chat` - Send message with session support
- `GET /api/health` - Health check

## 📖 Documentation

- **[Deployment Guide](DEPLOYMENT.md)** - Complete deployment instructions
- **[API Documentation](http://localhost:8080/docs)** - Interactive API docs

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  F1 App (FastAPI) │    │  PostgreSQL DB  │
│   (Production)  │◄──►│  + Session Mgmt │◄──►│  + F1 Data      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.