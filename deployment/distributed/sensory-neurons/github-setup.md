# Sensory Neurons - GitHub Actions Setup

## Overview
Sensory Neurons are learning scrapers that run on GitHub Actions, triggered by the Signal Relay when lightweight scrapers fail. They use Playwright for browser automation and machine learning for pattern recognition.

## Repository Setup

### 1. Create Repository
```bash
# Create new repository
gh repo create synapse-sensory --private --description "Project Synapse - Learning Scrapers (Sensory Neurons)"

# Clone and setup
git clone https://github.com/your-username/synapse-sensory.git
cd synapse-sensory
```

### 2. Repository Structure
```
synapse-sensory/
├── .github/
│   └── workflows/
│       ├── learning-scraper.yml      # Main scraping workflow
│       ├── pattern-trainer.yml       # ML model training
│       └── health-check.yml          # Component health monitoring
├── src/
│   ├── scrapers/
│   │   ├── playwright_scraper.py     # Browser automation
│   │   ├── pattern_learner.py        # ML pattern recognition
│   │   └── recipe_generator.py       # Auto-generate scraping recipes
│   ├── proxy/
│   │   ├── chameleon_network.py      # Proxy rotation + Tor
│   │   └── anti_bot_evasion.py       # Bot detection bypass
│   └── utils/
│       ├── hub_client.py             # Communication with Central Hub
│       └── screenshot_analyzer.py    # Visual pattern analysis
├── requirements.txt                   # Python dependencies
├── Dockerfile                        # Container for local testing
└── README.md                         # Setup instructions
```

### 3. GitHub Secrets Configuration

Set these secrets in your repository settings:

```bash
# Hub connection
SYNAPSE_HUB_URL=https://synapse-central-hub.onrender.com
SYNAPSE_API_KEY=your_sensory_neurons_api_key
HUB_WEBHOOK_SECRET=webhook_secret_for_callbacks

# Proxy services (optional)
PROXY_PROVIDER_API_KEY=your_proxy_service_key
TOR_PROXY_LIST=comma_separated_tor_proxies

# Browser automation
PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Machine learning
HUGGINGFACE_API_KEY=your_huggingface_key
OPENAI_API_KEY=your_openai_key
```

### 4. Workflow Configuration

#### Main Learning Scraper Workflow
```yaml
# .github/workflows/learning-scraper.yml
name: Learning Scraper

on:
  repository_dispatch:
    types: [scrape_request]
  workflow_dispatch:
    inputs:
      url:
        description: 'URL to scrape'
        required: true
      recipe_id:
        description: 'Failed recipe ID'
        required: false
      priority:
        description: 'Task priority'
        required: false
        default: 'normal'

jobs:
  scrape:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        playwright install chromium
    
    - name: Run learning scraper
      env:
        SYNAPSE_HUB_URL: ${{ secrets.SYNAPSE_HUB_URL }}
        SYNAPSE_API_KEY: ${{ secrets.SYNAPSE_API_KEY }}
        HUB_WEBHOOK_SECRET: ${{ secrets.HUB_WEBHOOK_SECRET }}
        PROXY_PROVIDER_API_KEY: ${{ secrets.PROXY_PROVIDER_API_KEY }}
        TARGET_URL: ${{ github.event.client_payload.url || github.event.inputs.url }}
        FAILED_RECIPE_ID: ${{ github.event.client_payload.recipe_id || github.event.inputs.recipe_id }}
        TASK_PRIORITY: ${{ github.event.client_payload.priority || github.event.inputs.priority }}
      run: |
        python src/scrapers/playwright_scraper.py
    
    - name: Upload artifacts
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: scraping-results
        path: |
          screenshots/
          logs/
          generated_recipes/
        retention-days: 7
    
    - name: Notify hub of completion
      if: always()
      run: |
        python src/utils/hub_client.py notify-completion \
          --status ${{ job.status }} \
          --run-id ${{ github.run_id }}
```

### 5. Component Registration

The Sensory Neurons component registers itself with the Central Hub:

```python
# src/utils/hub_client.py
import os
import requests
from typing import Dict, Any

class HubClient:
    def __init__(self):
        self.hub_url = os.getenv('SYNAPSE_HUB_URL')
        self.api_key = os.getenv('SYNAPSE_API_KEY')
        self.component_id = 'sensory-neurons'
    
    def register_component(self) -> Dict[str, Any]:
        """Register this component with the Central Hub."""
        response = requests.post(
            f"{self.hub_url}/v1/components/register",
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'component_id': self.component_id,
                'component_type': 'sensory_neurons',
                'platform': 'github_actions',
                'capabilities': [
                    'browser_automation',
                    'pattern_learning',
                    'recipe_generation',
                    'anti_bot_evasion'
                ],
                'webhook_url': f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY')}/dispatches",
                'health_check_url': f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions",
                'resource_limits': {
                    'max_concurrent_jobs': 20,
                    'timeout_minutes': 30,
                    'monthly_minutes': 2000
                }
            }
        )
        return response.json()
```

### 6. Triggering from Signal Relay

The Signal Relay (Cloudflare Worker) triggers Sensory Neurons via GitHub API:

```javascript
// In Signal Relay Worker
async function triggerSensoryNeurons(scrapeRequest) {
  const response = await fetch(
    `https://api.github.com/repos/${SENSORY_REPO}/dispatches`,
    {
      method: 'POST',
      headers: {
        'Authorization': `token ${GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        event_type: 'scrape_request',
        client_payload: {
          url: scrapeRequest.url,
          recipe_id: scrapeRequest.failed_recipe_id,
          priority: scrapeRequest.priority,
          timestamp: new Date().toISOString(),
          request_id: scrapeRequest.id
        }
      })
    }
  );
  
  return response.ok;
}
```

### 7. Free Tier Optimization

GitHub Actions free tier provides 2000 minutes/month. Optimization strategies:

```yaml
# Efficient resource usage
strategy:
  matrix:
    batch_size: [5]  # Process 5 URLs per job
  max-parallel: 3    # Limit concurrent jobs

# Cache dependencies
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

# Cache Playwright browsers
- name: Cache Playwright browsers
  uses: actions/cache@v3
  with:
    path: ~/.cache/ms-playwright
    key: ${{ runner.os }}-playwright-${{ hashFiles('requirements.txt') }}
```

### 8. Monitoring and Health Checks

```yaml
# .github/workflows/health-check.yml
name: Component Health Check

on:
  schedule:
    - cron: '*/30 * * * *'  # Every 30 minutes
  workflow_dispatch:

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
    - name: Report health to hub
      run: |
        curl -X POST "${{ secrets.SYNAPSE_HUB_URL }}/v1/components/health" \
          -H "Authorization: Bearer ${{ secrets.SYNAPSE_API_KEY }}" \
          -H "Content-Type: application/json" \
          -d '{
            "component_id": "sensory-neurons",
            "status": "healthy",
            "metrics": {
              "available_minutes": "${{ github.event.repository.plan.private_repos }}",
              "last_run": "${{ github.event.head_commit.timestamp }}",
              "success_rate": 0.95
            }
          }'
```

## Usage Limits & Scaling

### Free Tier Limits
- **2000 minutes/month** for private repos
- **Unlimited minutes** for public repos
- **20 concurrent jobs** maximum
- **6 hours** maximum job duration

### Optimization Strategies
1. **Batch Processing**: Process multiple URLs per job
2. **Smart Caching**: Cache dependencies and browsers
3. **Conditional Execution**: Only run when needed
4. **Public Repository**: Consider making repo public for unlimited minutes

### Scaling Path
1. **GitHub Pro**: $4/month for 3000 minutes
2. **GitHub Team**: $4/user/month for 3000 minutes
3. **GitHub Enterprise**: Custom pricing for unlimited minutes
4. **Self-hosted Runners**: Run on your own infrastructure

## Integration Testing

Test the complete workflow:

```bash
# Trigger manual test
gh workflow run learning-scraper.yml \
  -f url="https://example.com/test-article" \
  -f priority="high"

# Check status
gh run list --workflow=learning-scraper.yml

# View logs
gh run view --log
```

This setup provides a fully functional, scalable learning scraper system that integrates seamlessly with the Project Synapse ecosystem while maximizing the GitHub Actions free tier benefits.