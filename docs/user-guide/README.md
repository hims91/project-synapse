# Project Synapse User Guide

Welcome to Project Synapse - the definitive AI-powered content intelligence platform. This guide will help you get started and make the most of Synapse's powerful features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Content Analysis](#content-analysis)
4. [Semantic Search](#semantic-search)
5. [Web Scraping](#web-scraping)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Trends Analysis](#trends-analysis)
8. [API Integration](#api-integration)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Creating Your Account

1. Visit [projectsynapse.com](https://projectsynapse.com)
2. Click "Sign Up" and choose your plan:
   - **Free**: 1,000 API calls/day, basic features
   - **Pro**: 50,000 API calls/day, advanced analytics
   - **Enterprise**: Unlimited calls, custom features

3. Verify your email address
4. Complete your profile setup

### Getting Your API Key

1. Log in to your [dashboard](https://dashboard.projectsynapse.com)
2. Navigate to **Settings** → **API Keys**
3. Click **Generate New API Key**
4. Copy and securely store your API key
5. Set up your development environment

```bash
# Install the Python SDK
pip install synapse-api

# Or use npm for JavaScript
npm install @synapse/api
```

### Your First API Call

```python
import synapse

# Initialize the client
client = synapse.Client(api_key='your_api_key_here')

# Analyze a web page
result = client.content.analyze(
    url='https://techcrunch.com/latest-ai-article',
    analysis_types=['sentiment', 'bias', 'topics']
)

print(f"Sentiment: {result.sentiment.label}")
print(f"Bias Score: {result.bias.overall_score}")
print(f"Topics: {[topic.name for topic in result.topics]}")
```

## Dashboard Overview

### Main Navigation

The Synapse dashboard is organized into several key sections:

- **🏠 Home**: Overview and quick stats
- **📊 Analytics**: Content analysis and insights
- **🔍 Search**: Semantic search interface
- **🕷️ Scraping**: Web scraping management
- **📈 Trends**: Trend analysis and monitoring
- **🔔 Alerts**: Monitoring and notifications
- **⚙️ Settings**: Account and API configuration

### Quick Stats Widget

The home dashboard shows:
- **API Usage**: Current month's API calls
- **Content Analyzed**: Total articles processed
- **Active Monitors**: Running monitoring subscriptions
- **System Health**: Real-time system status

### Recent Activity

View your recent:
- Content analysis requests
- Search queries
- Scraping jobs
- Alert notifications

## Content Analysis

### Analyzing Web Content

#### Using the Dashboard

1. Navigate to **Analytics** → **Content Analysis**
2. Enter a URL or paste text content
3. Select analysis types:
   - **Sentiment Analysis**: Positive, negative, neutral
   - **Bias Detection**: Political, cultural, gender bias
   - **Topic Extraction**: Key themes and subjects
   - **Entity Recognition**: People, places, organizations
   - **Summarization**: Key points and summary

4. Click **Analyze Content**
5. View results in real-time

#### Using the API

```python
# Analyze a news article
result = client.content.analyze(
    url='https://example.com/news-article',
    analysis_types=['sentiment', 'bias', 'topics', 'entities', 'summary']
)

# Access results
print(f"Sentiment: {result.sentiment.label} ({result.sentiment.score:.2f})")
print(f"Bias Score: {result.bias.overall_score:.2f}")

for topic in result.topics:
    print(f"Topic: {topic.name} (confidence: {topic.confidence:.2f})")

for entity in result.entities:
    print(f"Entity: {entity.text} ({entity.type})")

print(f"Summary: {result.summary.text}")
```

### Understanding Analysis Results

#### Sentiment Analysis
- **Score Range**: -1.0 (very negative) to +1.0 (very positive)
- **Labels**: negative, neutral, positive
- **Confidence**: How certain the model is (0.0 to 1.0)

#### Bias Detection
- **Overall Score**: 0.0 (no bias) to 1.0 (high bias)
- **Bias Types**:
  - Political: Left/right political leaning
  - Cultural: Cultural stereotypes or assumptions
  - Gender: Gender-based bias
  - Corporate: Pro/anti-business sentiment

#### Topic Extraction
- **Confidence**: How relevant the topic is (0.0 to 1.0)
- **Keywords**: Related terms and phrases
- **Hierarchical**: Topics can have sub-topics

## Semantic Search

### Basic Search

#### Dashboard Search

1. Go to **Search** in the main navigation
2. Enter your search query in natural language
3. Apply filters:
   - **Date Range**: When content was published
   - **Sentiment**: Positive, negative, neutral
   - **Topics**: Filter by specific topics
   - **Sources**: Limit to specific domains
   - **Language**: Content language

4. View results with relevance scores

#### API Search

```python
# Basic semantic search
results = client.search.query(
    query='artificial intelligence in healthcare',
    limit=10,
    filters={
        'sentiment': ['positive'],
        'topics': ['healthcare', 'technology'],
        'date_range': {
            'start': '2024-01-01',
            'end': '2024-01-08'
        }
    }
)

for result in results.results:
    print(f"Title: {result.title}")
    print(f"Relevance: {result.relevance_score:.2f}")
    print(f"URL: {result.url}")
    print("---")
```

### Advanced Search Features

#### Similar Content Discovery

```python
# Find content similar to a specific article
similar = client.search.similar(
    content_id='article_123',
    limit=5,
    similarity_threshold=0.7
)

for item in similar.similar_content:
    print(f"Similar: {item.title} (similarity: {item.similarity_score:.2f})")
```

#### Search Suggestions

```python
# Get search suggestions
suggestions = client.search.suggestions(
    query='artificial intel',
    limit=5
)

for suggestion in suggestions.suggestions:
    print(f"Suggestion: {suggestion.text}")
```

## Web Scraping

### One-Click Scraping

#### Dashboard Interface

1. Navigate to **Scraping** → **ScrapeDrop**
2. Enter the URL you want to scrape
3. Select analysis options:
   - Content extraction
   - Sentiment analysis
   - Topic detection
   - Bias analysis

4. Choose priority level:
   - **Low**: Process when resources available
   - **Normal**: Standard processing queue
   - **High**: Priority processing
   - **Critical**: Immediate processing

5. Click **Submit Job**
6. Monitor progress in real-time

#### API Scraping

```python
# Submit a scraping job
job = client.scraping.submit(
    url='https://example.com/article-to-scrape',
    analysis_types=['sentiment', 'topics', 'bias'],
    priority='normal'
)

print(f"Job ID: {job.job_id}")
print(f"Status: {job.status}")

# Check job status
status = client.scraping.status(job.job_id)
print(f"Progress: {status.progress * 100:.1f}%")

# Get results when complete
if status.status == 'completed':
    result = client.scraping.result(job.job_id)
    print(f"Title: {result.content.title}")
    print(f"Sentiment: {result.analysis.sentiment.label}")
```

### Batch Scraping

```python
# Submit multiple URLs for scraping
urls = [
    'https://example.com/article1',
    'https://example.com/article2',
    'https://example.com/article3'
]

batch_job = client.scraping.batch_submit(
    urls=urls,
    analysis_types=['sentiment', 'topics'],
    priority='normal'
)

print(f"Batch ID: {batch_job.batch_id}")

# Monitor batch progress
batch_status = client.scraping.batch_status(batch_job.batch_id)
print(f"Completed: {batch_status.completed_count}/{batch_status.total_count}")
```

### Scraping Recipes

Create custom scraping rules for specific websites:

```python
# Create a scraping recipe
recipe = client.scraping.create_recipe(
    name='TechCrunch Articles',
    domain='techcrunch.com',
    selectors={
        'title': 'h1.article-title',
        'content': '.article-content',
        'author': '.author-name',
        'date': '.publish-date'
    },
    test_url='https://techcrunch.com/sample-article'
)

# Use recipe for scraping
job = client.scraping.submit(
    url='https://techcrunch.com/new-article',
    recipe_id=recipe.recipe_id
)
```

## Monitoring & Alerts

### Setting Up Monitors

#### Keyword Monitoring

1. Go to **Alerts** → **Create Monitor**
2. Choose **Keyword Monitor**
3. Configure settings:
   - **Name**: Descriptive name for your monitor
   - **Keywords**: Terms to monitor
   - **Sources**: Websites, feeds, or social media
   - **Frequency**: How often to check
   - **Filters**: Sentiment, language, etc.

4. Set up notifications:
   - **Email**: Get alerts via email
   - **Webhook**: Send to your application
   - **Slack**: Post to Slack channel
   - **Discord**: Send to Discord server

#### Content Change Monitoring

```python
# Monitor a webpage for changes
monitor = client.monitoring.create_subscription(
    name='Competitor Product Page',
    url='https://competitor.com/products',
    monitor_type='content_change',
    frequency='hourly',
    webhook_url='https://yourapp.com/webhook',
    filters={
        'min_change_threshold': 0.1,  # 10% content change
        'ignore_elements': ['.timestamp', '.ads']
    }
)

print(f"Monitor ID: {monitor.subscription_id}")
```

### Managing Alerts

#### Dashboard Management

1. View all monitors in **Alerts** → **My Monitors**
2. See recent alerts and their status
3. Edit monitor settings
4. Pause/resume monitors
5. View alert history and analytics

#### API Management

```python
# List all monitors
monitors = client.monitoring.list_subscriptions()

for monitor in monitors.subscriptions:
    print(f"Monitor: {monitor.name} (Status: {monitor.status})")

# Update a monitor
client.monitoring.update_subscription(
    subscription_id='monitor_123',
    frequency='daily',
    active=True
)

# Delete a monitor
client.monitoring.delete_subscription('monitor_123')
```

## Trends Analysis

### Discovering Trends

#### Real-time Trends

1. Navigate to **Trends** → **Trending Now**
2. View current trending topics
3. Filter by:
   - Time range (1h, 6h, 24h, 7d, 30d)
   - Category (technology, politics, business, etc.)
   - Geographic region
   - Language

4. Click on trends for detailed analysis

#### API Trends

```python
# Get trending topics
trends = client.trends.get_trending_topics(
    time_range='24h',
    limit=10,
    category='technology'
)

for trend in trends.trending_topics:
    print(f"Topic: {trend.topic}")
    print(f"Mentions: {trend.mentions}")
    print(f"Growth Rate: {trend.growth_rate:.2f}")
    print(f"Sentiment: {trend.sentiment.overall:.2f}")
    print("---")
```

### Custom Trend Analysis

```python
# Analyze trends for specific topics
analysis = client.trends.analyze(
    topics=['artificial intelligence', 'machine learning'],
    time_range={
        'start': '2024-01-01T00:00:00Z',
        'end': '2024-01-08T23:59:59Z'
    },
    metrics=['volume', 'sentiment', 'geographic']
)

for topic, data in analysis.results.items():
    print(f"Topic: {topic}")
    print(f"Volume Trend: {data.volume_trend.trend_direction}")
    print(f"Sentiment Trend: {data.sentiment_trend.average_sentiment:.2f}")
```

### Trend Predictions

```python
# Get trend predictions
predictions = client.trends.predict(
    topic='artificial intelligence',
    prediction_horizon='7d',
    confidence_level=0.8
)

print(f"Predicted Volume: {predictions.volume.forecast[0].predicted_mentions}")
print(f"Confidence: {predictions.volume.confidence:.2f}")
print(f"Trend Direction: {predictions.volume.trend_direction}")
```

## API Integration

### Authentication

All API requests require authentication using your API key:

```python
# Python SDK
import synapse
client = synapse.Client(api_key='your_api_key')

# Direct HTTP requests
import requests

headers = {
    'Authorization': 'Bearer your_api_key',
    'Content-Type': 'application/json'
}

response = requests.get(
    'https://api.projectsynapse.com/v1/health',
    headers=headers
)
```

### Rate Limiting

Monitor your API usage to avoid rate limits:

```python
# Check rate limit status
status = client.account.rate_limit_status()

print(f"Remaining: {status.remaining}")
print(f"Reset Time: {status.reset_time}")
print(f"Limit: {status.limit}")
```

### Error Handling

```python
try:
    result = client.content.analyze(url='https://example.com/article')
except synapse.RateLimitError as e:
    print(f"Rate limit exceeded. Reset at: {e.reset_time}")
except synapse.AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
except synapse.ValidationError as e:
    print(f"Invalid request: {e.details}")
except synapse.APIError as e:
    print(f"API error: {e.message}")
```

### Webhooks

Set up webhooks to receive real-time notifications:

```python
# Create a webhook endpoint
webhook = client.webhooks.create(
    url='https://yourapp.com/synapse-webhook',
    events=['scraping.job.completed', 'monitoring.alert.triggered'],
    secret='your_webhook_secret'
)

# Verify webhook signatures in your app
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## Advanced Features

### Custom Analysis Models

Train custom models for your specific use case:

```python
# Create a custom sentiment model
model = client.models.create_custom_sentiment(
    name='Financial News Sentiment',
    training_data=[
        {'text': 'Stock prices soared today', 'sentiment': 'positive'},
        {'text': 'Market crash imminent', 'sentiment': 'negative'},
        # ... more training examples
    ]
)

# Use custom model for analysis
result = client.content.analyze(
    text='The quarterly earnings exceeded expectations',
    models={'sentiment': model.model_id}
)
```

### Bulk Operations

Process large amounts of content efficiently:

```python
# Bulk content analysis
results = client.content.bulk_analyze(
    items=[
        {'url': 'https://example.com/article1'},
        {'url': 'https://example.com/article2'},
        {'text': 'Direct text content to analyze'},
    ],
    analysis_types=['sentiment', 'topics'],
    callback_url='https://yourapp.com/bulk-complete'
)

print(f"Bulk Job ID: {results.bulk_id}")
```

### Data Export

Export your analyzed content:

```python
# Export content analysis results
export = client.data.export(
    format='json',  # or 'csv', 'xlsx'
    date_range={
        'start': '2024-01-01',
        'end': '2024-01-08'
    },
    filters={
        'sentiment': ['positive'],
        'topics': ['technology']
    }
)

# Download export file
with open('synapse_export.json', 'wb') as f:
    f.write(export.download())
```

## Troubleshooting

### Common Issues

#### API Key Problems

**Issue**: "Invalid API key" error
**Solution**: 
1. Check that your API key is correct
2. Ensure you're using the right environment (staging vs production)
3. Verify your account is active and not suspended

#### Rate Limiting

**Issue**: "Rate limit exceeded" error
**Solution**:
1. Check your current usage in the dashboard
2. Implement exponential backoff in your code
3. Consider upgrading your plan for higher limits

#### Slow Response Times

**Issue**: API requests taking too long
**Solution**:
1. Check the API status page
2. Reduce the complexity of your analysis requests
3. Use batch operations for multiple items
4. Implement caching in your application

#### Webhook Issues

**Issue**: Webhooks not being received
**Solution**:
1. Verify your webhook URL is accessible
2. Check webhook signature verification
3. Ensure your endpoint returns 200 status
4. Check webhook logs in the dashboard

### Getting Help

#### Self-Service Resources

1. **Documentation**: [docs.projectsynapse.com](https://docs.projectsynapse.com)
2. **API Status**: [status.projectsynapse.com](https://status.projectsynapse.com)
3. **FAQ**: [projectsynapse.com/faq](https://projectsynapse.com/faq)
4. **Community Forum**: [community.projectsynapse.com](https://community.projectsynapse.com)

#### Contact Support

- **Email**: [support@projectsynapse.com](mailto:support@projectsynapse.com)
- **Discord**: [discord.gg/projectsynapse](https://discord.gg/projectsynapse)
- **Live Chat**: Available in the dashboard (Pro+ plans)
- **Phone Support**: Available for Enterprise customers

#### Bug Reports

Report bugs on GitHub: [github.com/project-synapse/synapse/issues](https://github.com/project-synapse/synapse/issues)

Include:
- Your API key (first 8 characters only)
- Request/response details
- Error messages
- Steps to reproduce

---

## Next Steps

Now that you're familiar with Project Synapse, explore these advanced topics:

- [Developer Guide](../developer-guide/README.md) - Deep dive into API integration
- [Best Practices](../best-practices/README.md) - Optimization tips and patterns
- [Use Cases](../use-cases/README.md) - Real-world implementation examples
- [API Reference](../api/reference/README.md) - Complete API documentation

Happy analyzing! 🧠✨