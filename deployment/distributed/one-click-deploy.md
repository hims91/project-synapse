# 🚀 Project Synapse - One-Click Distributed Deployment

## 🎯 Complete Deployment in 6 Clicks

Deploy the entire Project Synapse brain-inspired architecture across multiple free-tier platforms with just 6 clicks!

---

## 🧠 Step 1: Deploy Central Hub (Render)
**The main brain that coordinates all components**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/project-synapse/synapse)

**What this deploys:**
- Central Cortex (FastAPI Hub)
- Thalamus (NLP Engine)
- Synaptic Vesicle (PostgreSQL Database)
- Main API endpoints and WebSocket server

**Free tier includes:** 750 hours/month (24/7 operation)

---

## 🌐 Step 2: Deploy Dendrites (Cloudflare Workers)
**High-frequency feed polling system**

```bash
# One command deployment
npx wrangler deploy --env production
```

Or use the web interface:
[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/project-synapse/synapse-dendrites)

**What this deploys:**
- RSS/Atom feed pollers
- Priority-based scheduling
- Cloudflare KV storage
- Cron triggers for automation

**Free tier includes:** 100,000 requests/day

---

## 🕷️ Step 3: Deploy Neurons (Railway)
**Lightweight scraping workers**

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/synapse-neurons)

**What this deploys:**
- HTTP-based scrapers
- Recipe-based scraping engine
- Docker containers with auto-scaling
- Health monitoring and metrics

**Free tier includes:** 500 hours/month

---

## 🤖 Step 4: Deploy Sensory Neurons (GitHub Actions)
**Learning scrapers with browser automation**

1. Fork the repository: [synapse-sensory](https://github.com/project-synapse/synapse-sensory)
2. Enable GitHub Actions in repository settings
3. Set required secrets (see setup guide)

**What this deploys:**
- Playwright browser automation
- Machine learning pattern recognition
- Anti-bot evasion with proxy rotation
- Automatic recipe generation

**Free tier includes:** 2,000 minutes/month

---

## 🦴 Step 5: Deploy Spinal Cord (Vercel + Cloudflare R2)
**Fallback system for resilience**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/project-synapse/synapse-spinal)

**What this deploys:**
- Fallback task storage
- Automatic recovery system
- Serverless functions
- Health monitoring

**Free tier includes:** 100GB bandwidth/month

---

## 🎨 Step 6: Deploy Frontend (Netlify)
**Interactive dashboard and UI**

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/project-synapse/synapse-frontend)

**What this deploys:**
- React-based dashboard
- Real-time data visualization
- API testing interface
- User management system

**Free tier includes:** 100GB bandwidth/month

---

## ⚙️ Post-Deployment Configuration

### 1. Get Your Hub URL
After Step 1 completes, note your Render app URL:
```
https://your-app-name.onrender.com
```

### 2. Configure Component Connections
Each component needs to know how to reach the hub. Set these environment variables:

**For all components:**
```env
SYNAPSE_HUB_URL=https://your-app-name.onrender.com
SYNAPSE_API_KEY=your_component_api_key
```

### 3. Generate API Keys
The Central Hub automatically generates API keys for each component. Access them via:
```bash
curl https://your-app-name.onrender.com/v1/admin/api-keys \
  -H "Authorization: Bearer your_admin_key"
```

### 4. Verify Integration
Check that all components are connected:
```bash
curl https://your-app-name.onrender.com/v1/system/status
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "central_cortex": "healthy",
    "dendrites": "healthy",
    "neurons": "healthy",
    "sensory_neurons": "healthy",
    "spinal_cord": "healthy",
    "frontend": "healthy"
  },
  "total_components": 6,
  "healthy_components": 6
}
```

---

## 🎛️ Environment Variables Reference

### Central Hub (Render)
```env
# Required
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
SECRET_KEY=your_secret_key

# Component API Keys (auto-generated)
DENDRITES_API_KEY=dendrites_key
NEURONS_API_KEY=neurons_key
SENSORY_API_KEY=sensory_key
SPINAL_API_KEY=spinal_key

# External Services
CLOUDFLARE_API_TOKEN=your_token
CLOUDFLARE_ACCOUNT_ID=your_account_id
```

### Dendrites (Cloudflare Workers)
```env
# Secrets (set via wrangler secret put)
SYNAPSE_API_KEY=dendrites_key
HUB_WEBHOOK_SECRET=webhook_secret
```

### Neurons (Railway)
```env
SYNAPSE_HUB_URL=https://your-app-name.onrender.com
SYNAPSE_API_KEY=neurons_key
HUB_WEBHOOK_SECRET=webhook_secret
```

### Sensory Neurons (GitHub Secrets)
```env
SYNAPSE_HUB_URL=https://your-app-name.onrender.com
SYNAPSE_API_KEY=sensory_key
HUB_WEBHOOK_SECRET=webhook_secret
PROXY_PROVIDER_API_KEY=optional_proxy_key
```

### Spinal Cord (Vercel)
```env
SYNAPSE_HUB_URL=https://your-app-name.onrender.com
SYNAPSE_API_KEY=spinal_key
CLOUDFLARE_R2_BUCKET=synapse-fallback-storage
```

### Frontend (Netlify)
```env
REACT_APP_API_URL=https://your-app-name.onrender.com
REACT_APP_WS_URL=wss://your-app-name.onrender.com/ws
```

---

## 📊 Cost Breakdown (All Free Tiers)

| Component | Platform | Monthly Cost | Usage Limits |
|-----------|----------|--------------|--------------|
| Central Hub | Render | $0 | 750 hours (24/7) |
| Dendrites | Cloudflare | $0 | 100k requests/day |
| Neurons | Railway | $0 | 500 hours |
| Sensory Neurons | GitHub | $0 | 2000 minutes |
| Spinal Cord | Vercel | $0 | 100GB bandwidth |
| Frontend | Netlify | $0 | 100GB bandwidth |
| **Total** | **$0** | **Generous limits** |

**Estimated capacity:** 
- 1M+ API requests/month
- 10k+ scraping jobs/month
- 24/7 uptime
- Global CDN distribution

---

## 🔄 Automated Deployment Script

For advanced users, deploy everything with one script:

```bash
#!/bin/bash
# deploy-synapse.sh - One-click deployment script

echo "🧠 Deploying Project Synapse..."

# 1. Deploy Central Hub
echo "1/6 Deploying Central Hub to Render..."
curl -X POST "https://api.render.com/v1/services" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -d @deployment/distributed/central-hub/render.yaml

# 2. Deploy Dendrites
echo "2/6 Deploying Dendrites to Cloudflare..."
cd deployment/distributed/dendrites
wrangler deploy --env production

# 3. Deploy Neurons
echo "3/6 Deploying Neurons to Railway..."
railway deploy --service synapse-neurons

# 4. Setup Sensory Neurons
echo "4/6 Setting up Sensory Neurons on GitHub..."
gh repo create synapse-sensory --private
gh secret set SYNAPSE_HUB_URL --body "$HUB_URL"
gh secret set SYNAPSE_API_KEY --body "$SENSORY_API_KEY"

# 5. Deploy Spinal Cord
echo "5/6 Deploying Spinal Cord to Vercel..."
vercel deploy --prod

# 6. Deploy Frontend
echo "6/6 Deploying Frontend to Netlify..."
netlify deploy --prod --dir=frontend/build

echo "✅ Project Synapse deployed successfully!"
echo "🌐 Access your dashboard at: https://your-domain.netlify.app"
echo "🔗 API endpoint: https://your-app-name.onrender.com"
```

---

## 🚨 Troubleshooting

### Common Issues

**1. Components can't reach the Hub**
- Verify `SYNAPSE_HUB_URL` is set correctly
- Check API keys are properly configured
- Ensure Hub is fully deployed before other components

**2. Database connection errors**
- Verify Supabase database is created
- Check `DATABASE_URL` environment variable
- Ensure database migrations have run

**3. Cloudflare Workers not triggering**
- Verify cron triggers are enabled
- Check KV namespaces are bound correctly
- Ensure API keys have proper permissions

**4. GitHub Actions not running**
- Verify repository secrets are set
- Check workflow permissions
- Ensure Actions are enabled for the repository

### Health Check Commands

```bash
# Check Central Hub
curl https://your-app-name.onrender.com/health

# Check Dendrites
curl https://feeds.your-domain.com/health

# Check Neurons
curl https://synapse-neurons.railway.app/health

# Check Spinal Cord
curl https://synapse-spinal.vercel.app/health

# Check Frontend
curl https://your-domain.netlify.app
```

---

## 🎉 Success!

Once all components are deployed and connected, you'll have a fully distributed, brain-inspired AI system running across multiple cloud platforms at zero cost!

**Your Project Synapse network includes:**
- ✅ Real-time content analysis
- ✅ Intelligent web scraping
- ✅ Semantic search capabilities
- ✅ Trend detection and monitoring
- ✅ Automatic failover and recovery
- ✅ Interactive dashboard
- ✅ Complete API ecosystem

**Next steps:**
1. Explore the dashboard at your frontend URL
2. Test the API endpoints
3. Set up monitoring and alerts
4. Scale individual components as needed

Welcome to the future of distributed AI! 🚀🧠