# 🚀 Railway Deployment Guide

## Quick Deploy Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### 2. Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `formula-1-bot` repository
6. Railway will auto-detect your Docker setup

### 3. Add PostgreSQL Database
1. In Railway dashboard, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically connect it to your app

### 4. Configure Environment Variables
In Railway dashboard, add these variables:

**Required:**
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Optional:**
```
LANGSMITH_API_KEY=your_langsmith_key_here
LOG_LEVEL=INFO
```

**Database variables will be auto-provided by Railway**

### 5. Enable Auto-Deploy
1. Go to project settings
2. Enable "Auto Deploy"
3. Select your branch (usually `main`)

## What Happens During Deployment

1. **Build Phase** (5-10 minutes):
   - Railway builds your Docker image
   - Installs all dependencies
   - Sets up the container

2. **Startup Phase** (10-15 minutes):
   - Runs `./start.sh`
   - Waits for PostgreSQL to be ready
   - Runs `main.py` to populate database
   - Starts FastAPI app on Railway's PORT

3. **Health Check**:
   - Railway checks `/api/health` endpoint
   - App is ready when health check passes

## Monitoring

### View Logs
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and view logs
railway login
railway logs
```

### Check Status
```bash
railway status
```

### Open App
```bash
railway open
```

## Expected Timeline

- **First Deploy**: 15-20 minutes (includes DB population)
- **Subsequent Deploys**: 3-5 minutes
- **Auto-Deploy**: Triggers on every GitHub push

## Cost

- **Free Tier**: $5/month credit
- **Your App**: ~$3-4/month
- **Remaining**: $1-2/month for other projects

## Troubleshooting

### Common Issues

1. **Build Fails**: Check Dockerfile syntax
2. **Health Check Fails**: Check if app starts properly
3. **Database Connection**: Ensure PostgreSQL is added
4. **Environment Variables**: Verify all required keys are set

### Debug Commands
```bash
# View detailed logs
railway logs --tail

# Check environment variables
railway variables

# Restart deployment
railway service restart
```

## Your App URL

After successful deployment, Railway will provide a URL like:
```
https://formula-1-bot-production.up.railway.app
```

## Features Ready

✅ **Multi-user sessions**  
✅ **PostgreSQL database**  
✅ **Auto-deploy from GitHub**  
✅ **Health checks**  
✅ **Environment variables**  
✅ **Logging and monitoring**  

Your F1 bot is now ready for production! 🏎️ 