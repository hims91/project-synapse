# Central Hub Deployment Guide

## Prerequisites
- Supabase database connection string
- Redis (Upstash) connection URL
- GitHub account
- Render account

## Step 1: Connect GitHub Repository to Render

1. **Go to Render Dashboard:**
   - Open https://render.com
   - Sign in with your GitHub account

2. **Create New Web Service:**
   - Click "New +" button
   - Select "Web Service"
   - Choose "Build and deploy from a Git repository"
   - Click "Connect" next to your repository

3. **Configure Service:**
   - **Name**: `synapse-central-hub`
   - **Region**: Choose same as your Supabase (e.g., US East)
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Runtime**: `Docker`
   - **Dockerfile Path**: `deployment/distributed/central-hub/Dockerfile`

## Step 2: Configure Environment Variables

Add these environment variables in Render:

### Required Variables:
```
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=10000

# Database (from your Supabase setup)
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@db.abc123xyz.supabase.co:5432/postgres

# Redis (from your Upstash setup)
REDIS_URL=redis://default:your_password@region.upstash.io:6379

# Security (Render will generate these)
SECRET_KEY=[Generate Random Value]
JWT_SECRET_KEY=[Generate Random Value]
API_KEY_ENCRYPTION_KEY=[Generate Random Value]

# Component API Keys (Generate random values)
DENDRITES_API_KEY=[Generate Random Value]
NEURONS_API_KEY=[Generate Random Value]
SENSORY_API_KEY=[Generate Random Value]
SPINAL_API_KEY=[Generate Random Value]
```

### Optional Variables (for advanced features):
```
CLOUDFLARE_API_TOKEN=[Your Cloudflare Token]
CLOUDFLARE_ACCOUNT_ID=[Your Cloudflare Account ID]
CLOUDFLARE_R2_BUCKET=synapse-fallback-storage
```

## Step 3: Deploy

1. **Review Settings:**
   - Build Command: (leave empty - Docker handles this)
   - Start Command: (leave empty - Docker handles this)
   - Health Check Path: `/health`

2. **Deploy:**
   - Click "Create Web Service"
   - Wait for build and deployment (5-10 minutes)

3. **Verify Deployment:**
   - Once deployed, visit your service URL
   - You should see: `{"service": "Project Synapse - Central Cortex", "status": "operational"}`
   - Check `/health` endpoint for detailed health status

## Step 4: Save Your Hub URL

Your Central Hub will be available at:
```
https://synapse-central-hub.onrender.com
```

**Save this URL** - you'll need it for configuring other components!

## Troubleshooting

### Build Fails:
- Check that all files are committed to GitHub
- Verify Dockerfile path is correct
- Check build logs in Render dashboard

### Service Won't Start:
- Verify all environment variables are set
- Check that DATABASE_URL and REDIS_URL are correct
- Review service logs in Render dashboard

### Health Check Fails:
- Ensure port 10000 is exposed
- Verify the `/health` endpoint is accessible
- Check application logs for errors

## Next Steps

Once your Central Hub is deployed and healthy:
1. ✅ Save your hub URL
2. ✅ Test the `/health` endpoint
3. ✅ Move on to deploying the next component (Dendrites)