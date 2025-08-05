# Project Synapse API Reference

Welcome to the comprehensive API reference for Project Synapse - the definitive AI-powered content intelligence platform.

## Table of Contents

1. [Getting Started](../getting-started.md)
2. [Authentication](../authentication.md)
3. [Content Analysis API](content.md)
4. [Search API](search.md)
5. [Trends API](trends.md)
6. [Scraping API](scraping.md)
7. [Monitoring API](monitoring.md)
8. [Financial Intelligence API](financial.md)
9. [Summarization API](summarization.md)
10. [Bias Analysis API](bias.md)
11. [WebSocket API](websocket.md)
12. [Webhooks](../webhooks.md)

## API Overview

Project Synapse provides a comprehensive REST API with the following capabilities:

### 🧠 Core Intelligence Features
- **Content Analysis**: Deep semantic analysis with sentiment, bias, and topic detection
- **Semantic Search**: AI-powered search with relevance scoring and filtering
- **Trend Analysis**: Real-time trend detection and prediction
- **Bias Detection**: Comprehensive bias analysis across multiple dimensions

### 🕷️ Data Collection
- **Web Scraping**: Intelligent scraping with recipe learning
- **Feed Monitoring**: RSS/Atom feed polling with priority-based scheduling
- **URL Resolution**: Advanced URL decoding for redirectors and shortened links

### 📊 Specialized APIs
- **Financial Intelligence**: Market sentiment and trend analysis
- **Summarization**: Extractive and abstractive text summarization
- **Relationship Extraction**: Entity relationship mapping
- **Technical Intelligence**: Webpage metadata and technical analysis

### 🔄 Real-time Features
- **WebSocket API**: Real-time updates and notifications
- **Webhooks**: Event-driven integrations
- **Monitoring**: Keyword and content change monitoring

## Base URL

```
Production: https://api.projectsynapse.com
Staging: https://staging-api.projectsynapse.dev
```

## API Versioning

All API endpoints are versioned using URL path versioning:

```
https://api.projectsynapse.com/v1/
```

Current version: **v1**

## Content Types

The API accepts and returns JSON data:

```http
Content-Type: application/json
Accept: application/json
```

## Authentication

All API requests require authentication using API keys:

```http
Authorization: Bearer YOUR_API_KEY
```

See the [Authentication Guide](../authentication.md) for detailed information.

## Rate Limiting

API requests are rate-limited based on your subscription tier:

| Tier | Requests/minute | Requests/day | Concurrent |
|------|-----------------|--------------|------------|
| Free | 60 | 1,000 | 5 |
| Starter | 300 | 10,000 | 10 |
| Pro | 1,000 | 50,000 | 25 |
| Enterprise | 5,000 | 500,000 | 100 |

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1704708600
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is missing required parameters",
    "details": {
      "missing_fields": ["url"]
    },
    "request_id": "req_123456789",
    "timestamp": "2024-01-08T10:30:00Z"
  }
}
```

### Common Status Codes

- `200` - Success
- `201` - Created
- `202` - Accepted (for async operations)
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `502` - Bad Gateway
- `503` - Service Unavailable

## Pagination

List endpoints support pagination using query parameters:

```http
GET /v1/content/articles?page=2&per_page=20
```

Response includes pagination metadata:

```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "per_page": 20,
    "total": 1500,
    "pages": 75,
    "has_next": true,
    "has_prev": true
  }
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

```http
GET /v1/content/articles?sentiment=positive&topic=technology&sort=created_at&order=desc
```

Common filter parameters:
- `sentiment`: positive, negative, neutral
- `topic`: Filter by detected topics
- `date_range`: Date range filtering
- `language`: Content language
- `bias_score`: Bias score range

## Webhooks

Project Synapse supports webhooks for real-time event notifications:

```json
{
  "event": "scraping.job.completed",
  "data": {
    "job_id": "job_123456",
    "status": "completed",
    "result": {...}
  },
  "timestamp": "2024-01-08T10:30:00Z"
}
```

See the [Webhooks Guide](../webhooks.md) for setup instructions.

## WebSocket API

Real-time updates are available via WebSocket connections:

```javascript
const ws = new WebSocket('wss://api.projectsynapse.com/v1/ws');

ws.onopen = function() {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'YOUR_API_KEY'
  }));
};
```

See the [WebSocket API Reference](websocket.md) for details.

## SDKs and Libraries

Official SDKs are available for popular programming languages:

- **Python**: `pip install synapse-api`
- **JavaScript/Node.js**: `npm install @synapse/api`
- **Go**: `go get github.com/project-synapse/go-sdk`
- **Java**: Available on Maven Central
- **PHP**: `composer require synapse/api`
- **Ruby**: `gem install synapse-api`

## Code Examples

### Python

```python
import synapse

client = synapse.Client(api_key='your_api_key')

# Analyze content
result = client.content.analyze(
    url='https://example.com/article',
    analysis_types=['sentiment', 'bias', 'topics']
)

# Search content
results = client.search.query(
    query='artificial intelligence',
    filters={'sentiment': ['positive']}
)
```

### JavaScript

```javascript
const Synapse = require('@synapse/api');

const client = new Synapse.Client({
  apiKey: 'your_api_key'
});

// Analyze content
const result = await client.content.analyze({
  url: 'https://example.com/article',
  analysisTypes: ['sentiment', 'bias', 'topics']
});

// Search content
const results = await client.search.query({
  query: 'artificial intelligence',
  filters: { sentiment: ['positive'] }
});
```

### cURL

```bash
# Analyze content
curl -X POST https://api.projectsynapse.com/v1/content/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "analysis_types": ["sentiment", "bias", "topics"]
  }'

# Search content
curl -X POST https://api.projectsynapse.com/v1/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "filters": {"sentiment": ["positive"]}
  }'
```

## Support and Resources

- **Documentation**: [docs.projectsynapse.com](https://docs.projectsynapse.com)
- **API Status**: [status.projectsynapse.com](https://status.projectsynapse.com)
- **Support Email**: [support@projectsynapse.com](mailto:support@projectsynapse.com)
- **GitHub**: [github.com/project-synapse](https://github.com/project-synapse)
- **Discord Community**: [discord.gg/projectsynapse](https://discord.gg/projectsynapse)
- **Stack Overflow**: Tag questions with `project-synapse`

## Changelog

See the [API Changelog](../changelog.md) for version history and breaking changes.

## Terms of Service

By using the Project Synapse API, you agree to our [Terms of Service](https://projectsynapse.com/terms) and [Privacy Policy](https://projectsynapse.com/privacy).