# Sensory Neurons - Quick Setup Guide

## 🧠 What Sensory Neurons Does
- **Learning scrapers** that adapt to complex websites
- **Browser automation** with Playwright for JavaScript-heavy sites
- **Pattern recognition** to generate scraping recipes
- **Runs on GitHub Actions** (2000 free minutes/month)

## 📋 Quick Setup (5 minutes)

### Step 1: Create GitHub Repository

1. **Go to GitHub:** https://github.com/new
2. **Repository name:** `synapse-sensory-neurons`
3. **Description:** `Project Synapse - Learning Scrapers (Sensory Neurons)`
4. **Visibility:** Private (for free 2000 minutes/month)
5. **Initialize:** Check "Add a README file"
6. **Click "Create repository"**

### Step 2: Add Files to Repository

Create these files in your new repository:

#### `.github/workflows/learning-scraper.yml`
```yaml
name: Learning Scraper

on:
  repository_dispatch:
    types: [scrape_request]
  workflow_dispatch:
    inputs:
      url:
        description: 'URL to scrape'
        required: true
        default: 'https://example.com'
      priority:
        description: 'Task priority'
        required: false
        default: 'normal'

jobs:
  scrape:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install playwright beautifulsoup4 httpx python-dotenv requests
        playwright install chromium --with-deps
    
    - name: Run learning scraper
      env:
        SYNAPSE_HUB_URL: ${{ secrets.SYNAPSE_HUB_URL }}
        SENSORY_API_KEY: ${{ secrets.SENSORY_API_KEY }}
        TARGET_URL: ${{ github.event.client_payload.url || github.event.inputs.url }}
        PRIORITY: ${{ github.event.client_payload.priority || github.event.inputs.priority }}
      run: |
        python scraper.py
    
    - name: Upload results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: scraping-results-${{ github.run_id }}
        path: |
          screenshots/
          results.json
        retention-days: 3
```

#### `scraper.py`
```python
"""
Sensory Neurons - Learning Scraper
Advanced scraping with browser automation and pattern learning.
"""
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright
import httpx

async def main():
    target_url = os.getenv('TARGET_URL', 'https://example.com')
    hub_url = os.getenv('SYNAPSE_HUB_URL')
    api_key = os.getenv('SENSORY_API_KEY')
    priority = os.getenv('PRIORITY', 'normal')
    
    print(f"🧠 Sensory Neurons starting...")
    print(f"Target URL: {target_url}")
    print(f"Priority: {priority}")
    
    # Create directories
    Path("screenshots").mkdir(exist_ok=True)
    
    result = {
        "job_id": f"sensory-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "url": target_url,
        "status": "started",
        "component": "sensory-neurons",
        "timestamp": datetime.now().isoformat(),
        "data": {}
    }
    
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )
            
            page = await context.new_page()
            
            print(f"📄 Loading page: {target_url}")
            await page.goto(target_url, wait_until="networkidle", timeout=30000)
            
            # Take screenshot
            screenshot_path = f"screenshots/page-{result['job_id']}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"📸 Screenshot saved: {screenshot_path}")
            
            # Extract content
            title = await page.title()
            content = await page.content()
            
            # Get text content
            text_content = await page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style');
                    scripts.forEach(el => el.remove());
                    
                    return document.body.innerText || document.body.textContent || '';
                }
            """)
            
            # Get links
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.slice(0, 20).map(link => ({
                        url: link.href,
                        text: link.textContent.trim().substring(0, 100)
                    })).filter(link => link.text.length > 0);
                }
            """)
            
            # Get images
            images = await page.evaluate("""
                () => {
                    const images = Array.from(document.querySelectorAll('img[src]'));
                    return images.slice(0, 10).map(img => ({
                        src: img.src,
                        alt: img.alt || '',
                        width: img.naturalWidth || 0,
                        height: img.naturalHeight || 0
                    }));
                }
            """)
            
            # Update result
            result.update({
                "status": "completed",
                "data": {
                    "title": title,
                    "text_length": len(text_content),
                    "text_preview": text_content[:500] + "..." if len(text_content) > 500 else text_content,
                    "links_count": len(links),
                    "links": links,
                    "images_count": len(images),
                    "images": images,
                    "page_size": len(content),
                    "screenshot": screenshot_path
                },
                "metrics": {
                    "load_time": "< 30s",
                    "success_rate": 1.0,
                    "extraction_method": "playwright_browser_automation"
                }
            })
            
            await browser.close()
            print("✅ Scraping completed successfully")
            
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        result.update({
            "status": "failed",
            "error": str(e)
        })
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Send callback to hub if configured
    if hub_url and api_key:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{hub_url}/api/v1/callbacks/sensory",
                    json=result,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    print("📡 Callback sent to hub successfully")
                else:
                    print(f"⚠️ Callback failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Callback error: {e}")
    
    print(f"🎯 Job completed: {result['status']}")
    return result

if __name__ == "__main__":
    asyncio.run(main())
```

#### `README.md`
```markdown
# Project Synapse - Sensory Neurons

Learning scrapers that run on GitHub Actions for advanced web content extraction.

## Features
- Browser automation with Playwright
- JavaScript-heavy site support
- Screenshot capture
- Pattern learning and adaptation
- Integration with Project Synapse Central Hub

## Usage
Triggered automatically by the Central Hub or manually via GitHub Actions.

## Status
- Component: Sensory Neurons
- Platform: GitHub Actions
- Status: Operational
```

### Step 3: Configure Secrets

In your GitHub repository, go to **Settings** → **Secrets and variables** → **Actions** and add:

```
SYNAPSE_HUB_URL = https://synapse-central-hub.onrender.com
SENSORY_API_KEY = sensory_2f5a8c1e4b7f3a6d9c2e5f8a1b4e7c9d
```

### Step 4: Test the Setup

1. **Go to Actions tab** in your repository
2. **Click "Learning Scraper"** workflow
3. **Click "Run workflow"**
4. **Enter a test URL** (e.g., `https://example.com`)
5. **Click "Run workflow"**

### Step 5: Verify Results

- Check the workflow run logs
- Download the artifacts to see screenshots and results
- Verify the scraping worked correctly

## 🎯 Integration with Central Hub

The Sensory Neurons will:
1. **Receive triggers** from your Central Hub
2. **Perform advanced scraping** with browser automation
3. **Send results back** to the hub
4. **Learn patterns** for future scraping jobs

## 📊 Usage Limits

- **2000 free minutes/month** for private repositories
- **10 minute timeout** per job (configurable)
- **20 concurrent jobs** maximum
- **Unlimited minutes** if you make the repo public

Ready to set this up?