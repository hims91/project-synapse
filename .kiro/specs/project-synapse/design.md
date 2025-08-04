# Project Synapse Design Document

## Overview

Project Synapse is designed as a brain-inspired, multi-layer intelligence network that processes web content in real-time. The system follows a nervous system metaphor with specialized components handling different aspects of data flow, from initial perception through processing to final delivery via APIs.

The architecture prioritizes resilience, scalability, and cost-effectiveness by leveraging free-tier cloud services and implementing comprehensive fallback mechanisms. The system is designed for zero-downtime operation with automatic failover and self-healing capabilities.

## Architecture

### High-Level System Flow

```mermaid
graph TD
    A[Dendrites - Feed Pollers] --> B[Synaptic Vesicle - PostgreSQL Queue]
    B --> C[Neurons - Lightweight Scrapers]
    B --> D[Sensory Neurons - Learning Scrapers]
    C --> E[Signal Relay - Task Dispatcher]
    D --> E
    E --> F[Spinal Cord - Fallback Storage]
    E --> G[Central Cortex - Hub Server]
    G --> H[Thalamus - NLP Engine]
    H --> I[Axon Interface - Public APIs]
    F --> B
    
    subgraph "Layer 0: Sensory Input"
        A
    end
    
    subgraph "Layer 1: Perception"
        C
        D
    end
    
    subgraph "Layer 2: Signal Network"
        B
        E
        F
    end
    
    subgraph "Layer 3: Cerebral Cortex"
        G
        H
    end
    
    subgraph "Layer 4: Public Interface"
        I
    end
```

### Layer Architecture

#### Layer 0: Sensory Input Layer (Dendrites)
- **Technology**: Cloudflare Workers
- **Purpose**: High-frequency RSS/Atom feed polling
- **Design Pattern**: Priority-based polling with adaptive frequency
- **Scaling**: Automatic based on feed activity patterns
- **Resilience**: Built-in retry logic with exponential backoff

#### Layer 1: Perception Layer
- **Neurons (Lightweight Scrapers)**
  - Technology: Python + httpx, containerized with Docker
  - Design Pattern: Recipe-based scraping with 99%+ cache hit rate
  - Scaling: Horizontal scaling based on queue depth
  - Resilience: Automatic failover to learning mode on recipe failure

- **Sensory Neurons (Learning Scrapers)**
  - Technology: Python + Playwright, deployed on GitHub Actions
  - Design Pattern: Browser automation with ML-based pattern recognition
  - Scaling: On-demand execution triggered by scraping failures
  - Resilience: Chameleon Network (proxy rotation + Tor) for anti-bot bypass

#### Layer 2: Signal Network
- **Synaptic Vesicle (Primary Database)**
  - Technology: Supabase PostgreSQL with real-time subscriptions
  - Design Pattern: Event-driven architecture with WebSocket updates
  - Scaling: Connection pooling with read replicas
  - Resilience: Automatic failover to Spinal Cord during outages

- **Signal Relay (Task Dispatcher)**
  - Technology: Cloudflare Workers + Vercel Edge Functions (failover)
  - Design Pattern: Event-driven task distribution
  - Scaling: Serverless auto-scaling
  - Resilience: Multi-provider deployment with health checks

- **Spinal Cord (Fallback Queue)**
  - Technology: Cloudflare R2 object storage
  - Design Pattern: JSON-based task serialization
  - Scaling: Unlimited storage capacity
  - Resilience: Automatic re-injection when primary systems recover

#### Layer 3: Cerebral Cortex
- **Central Cortex (Hub Server)**
  - Technology: FastAPI on Render/Railway with auto-scaling
  - Design Pattern: Microservices with API gateway
  - Scaling: Container-based horizontal scaling
  - Resilience: Health checks with automatic restart

- **Thalamus (NLP Engine)**
  - Technology: Multi-tier processing (spaCy → TextRank → Hybrid models)
  - Design Pattern: Pipeline architecture with caching
  - Scaling: Async processing with worker queues
  - Resilience: Graceful degradation to simpler models

#### Layer 4: Public Interface (Axon Interface)
- **Technology**: RESTful APIs with OpenAPI specification
- **Design Pattern**: Freemium model with rate limiting
- **Scaling**: CDN caching with edge computing
- **Resilience**: Circuit breakers with fallback responses

## Components and Interfaces

### Core Data Models

#### Article Object Schema
```json
{
  "id": "uuid",
  "url": "string",
  "title": "string",
  "content": "string",
  "summary": "string",
  "author": "string",
  "publishedAt": "ISO8601",
  "scrapedAt": "ISO8601",
  "source": {
    "domain": "string",
    "name": "string",
    "category": "string"
  },
  "nlp": {
    "sentiment": "number",
    "entities": ["array"],
    "categories": ["array"],
    "significance": "number"
  },
  "metadata": {
    "paywall": "boolean",
    "canonicalUrl": "string",
    "techStack": ["array"]
  }
}
```

#### Scraping Recipe Schema
```json
{
  "id": "uuid",
  "domain": "string",
  "selectors": {
    "title": "string",
    "content": "string",
    "author": "string",
    "publishDate": "string"
  },
  "actions": ["array"],
  "success_rate": "number",
  "last_updated": "ISO8601",
  "created_by": "learning|manual"
}
```

### API Interface Specifications

#### Authentication System
- **Method**: Bearer token authentication
- **Rate Limiting**: Tier-based (1000 free calls/month)
- **Error Handling**: Standardized error response format
- **Monitoring**: Real-time usage tracking and alerting

#### Complete API Suite (Axon Interface)

1. **Content API** (`/content/*`)
   - `GET /content/articles` - Paginated article retrieval with filtering
   - `GET /content/articles/{id}` - Single article access
   - Query parameters: pageSize, page, sourceDomain, category, since

2. **Semantic Search API** (`/search`)
   - `GET /search` - Natural language search across data warehouse
   - Query parameters: q (query), limit
   - Returns relevance-scored results with _score field

3. **ScrapeDrop API** (`/scrape/*`)
   - `POST /scrape` - Submit URL for on-demand scraping
   - `GET /scrape/status/{jobId}` - Poll job status and results
   - Priority queue management with learning fallback

4. **WebWatch (Sentinel) API** (`/monitoring/*`)
   - `POST /monitoring/subscriptions` - Create keyword monitoring
   - `GET /monitoring/subscriptions` - List active subscriptions
   - `DELETE /monitoring/subscriptions/{id}` - Remove subscription
   - Webhook delivery with retry logic

5. **FinMind (Market Pulse) API** (`/financial/*`)
   - `GET /financial/market` - Financial news with sentiment analysis
   - Query parameters: tickers, sentiment
   - Specialized for financial content and market intelligence

6. **Digestify (Summarization) API** (`/summarize`)
   - `POST /summarize` - Generate article summaries
   - Request body: url, mode (extractive/abstractive)
   - Extractive (free tier) vs abstractive (premium) options

7. **InsightGraph (Signal Graph) API** (`/relationships`)
   - `GET /relationships` - Extract entity relationships from content
   - Query parameters: url
   - Returns subject-action-object relationship triplets

8. **MetaLens API** (`/meta`)
   - `GET /meta` - Technical webpage intelligence
   - Query parameters: url
   - Returns paywall status, canonical URLs, tech stack detection

9. **Chrono-Track API** (`/tracking/*`)
   - `POST /tracking/subscriptions` - Monitor webpage changes
   - Webhook notifications on content modifications
   - Visual diff capabilities for change detection

10. **Trends API** (`/trends`)
    - `GET /trends` - Real-time trending topics and entities
    - Query parameters: window (1h, 6h, 24h), category
    - Velocity and volume metrics for trending analysis

11. **Top Headlines API** (`/headlines`)
    - `GET /headlines` - Curated top headlines by category
    - Query parameters: category, limit
    - Significance scoring and ranking algorithm

12. **Bias & Narrative Analysis API** (`/analysis/narrative`)
    - `POST /analysis/narrative` - Advanced media bias analysis
    - Request body: query, articleIds (optional)
    - Sentiment aggregation, framing detection, narrative extraction

### Integration Points

#### External Service Dependencies
- **Cloudflare**: Workers, R2 storage, CDN
- **Supabase**: PostgreSQL, real-time subscriptions
- **Render/Railway**: Container hosting
- **GitHub Actions**: Learning scraper execution
- **Vercel**: Edge function failover

#### Internal Communication Protocols
- **HTTP/REST**: Primary API communication
- **WebSockets**: Real-time updates and notifications
- **Message Queues**: Async task processing
- **Event Streaming**: Cross-component notifications

## Data Models

### Database Schema Design

#### Core Tables
```sql
-- Articles table with full-text search
CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    author TEXT,
    published_at TIMESTAMPTZ,
    scraped_at TIMESTAMPTZ DEFAULT NOW(),
    source_domain TEXT NOT NULL,
    nlp_data JSONB,
    metadata JSONB,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED
);

-- Scraping recipes with success tracking
CREATE TABLE scraping_recipes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain TEXT UNIQUE NOT NULL,
    selectors JSONB NOT NULL,
    actions JSONB,
    success_rate DECIMAL(3,2) DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT 'learning'
);

-- Task queue with priority and retry logic
CREATE TABLE task_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    error_message TEXT
);

-- Monitoring subscriptions
CREATE TABLE monitoring_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    name TEXT NOT NULL,
    keywords TEXT[] NOT NULL,
    webhook_url TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_triggered TIMESTAMPTZ
);

-- API usage tracking
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
```

#### Indexing Strategy
- Full-text search indexes on article content
- Composite indexes for common query patterns
- Partial indexes for active monitoring subscriptions
- Time-series indexes for usage analytics

### Caching Strategy
- **Redis**: Hot data caching (trending topics, popular articles)
- **CDN**: Static content and API response caching
- **Application**: In-memory caching for recipes and configurations
- **Database**: Query result caching with invalidation triggers

## Error Handling

### Error Classification System

#### Error Types and Response Codes
1. **Authentication Errors (401)**
   - Invalid API key
   - Expired tokens
   - Rate limit exceeded

2. **Validation Errors (400)**
   - Invalid request parameters
   - Malformed JSON
   - Missing required fields

3. **Resource Errors (404/409)**
   - Article not found
   - Duplicate subscription
   - Invalid job ID

4. **System Errors (500/503)**
   - Database connectivity issues
   - External service failures
   - Processing timeouts

### Resilience Patterns

#### Circuit Breaker Implementation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
```

#### Retry Logic with Exponential Backoff
```python
async def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

### Fallback Mechanisms

#### Database Fallback Chain
1. **Primary**: Supabase PostgreSQL
2. **Secondary**: Local SQLite cache
3. **Tertiary**: Cloudflare R2 JSON storage
4. **Emergency**: In-memory storage with persistence

#### Service Degradation Levels
1. **Full Service**: All features operational
2. **Reduced Service**: Non-critical features disabled
3. **Essential Service**: Core APIs only
4. **Emergency Mode**: Cached responses only

## Testing Strategy

### Testing Pyramid

#### Unit Tests (70%)
- **Scope**: Individual functions and classes
- **Tools**: pytest, unittest.mock
- **Coverage**: >90% code coverage required
- **Focus**: Business logic, data transformations, utility functions

#### Integration Tests (20%)
- **Scope**: Component interactions and external services
- **Tools**: pytest-asyncio, testcontainers
- **Coverage**: API endpoints, database operations, external service calls
- **Focus**: Data flow between components

#### End-to-End Tests (10%)
- **Scope**: Complete user workflows
- **Tools**: Playwright, pytest-bdd
- **Coverage**: Critical user journeys
- **Focus**: API consumption patterns, dashboard functionality

### Test Data Management

#### Test Database Strategy
```python
# Fixture for test database
@pytest.fixture(scope="session")
async def test_db():
    # Create isolated test database
    test_db_url = "postgresql://test:test@localhost:5433/synapse_test"
    engine = create_async_engine(test_db_url)
    
    # Run migrations
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
```

#### Mock External Services
```python
# Mock Cloudflare Workers
@pytest.fixture
def mock_cloudflare_worker():
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"success": True}
        yield mock_post
```

### Performance Testing

#### Load Testing Strategy
- **Tool**: Locust for API load testing
- **Targets**: 1000 concurrent users, <200ms response time
- **Scenarios**: Normal usage, peak traffic, failure conditions
- **Metrics**: Throughput, latency, error rate, resource utilization

#### Monitoring and Alerting
- **Application Metrics**: Response times, error rates, queue depths
- **Infrastructure Metrics**: CPU, memory, disk, network usage
- **Business Metrics**: API usage, scraping success rates, user engagement
- **Alerting**: PagerDuty integration for critical issues

### Continuous Integration Pipeline

#### GitHub Actions Workflow
```yaml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          # Deployment commands
```

### UI/UX Design Considerations

#### Dashboard Features
- **System Health Widget**: Real-time monitoring of all components with nervous system visualization
- **Data Explorer**: Interactive table with expandable JSON views and "Copy Code Snippet" buttons (cURL, Python, JavaScript)
- **API Sandbox**: Live testing environment with instant code generation
- **Quick Start Guide**: Interactive onboarding for new developers
- **Usage Analytics**: API consumption tracking and billing information
- **One-Click Scrape Interface**: Large input box for URL analysis with real-time job status updates
- **Monitoring Management**: Visual webhook testing and subscription management
- **Trends Dashboard**: Real-time trending topics with clickable drill-down to search results
- **Market Pulse Charts**: Sentiment trends for financial tickers with article correlation
- **Narrative Analysis Workbench**: Interactive bias analysis with visual infographics

#### Developer Experience Features
- **Interactive Documentation**: Live API sandbox embedded in docs
- **Code Generation**: Automatic snippet generation for popular languages
- **Webhook Testing**: One-click webhook endpoint validation
- **Real-time Status**: WebSocket-powered live updates without page refresh
- **Visual Diff Viewer**: GitHub-style change detection for Chrono-Track
- **Relationship Visualizer**: Interactive subject-action-object diagrams

#### Performance Requirements
- **TTFB**: <150ms for all dashboard pages
- **API Response**: <200ms for 95th percentile
- **Search Latency**: <100ms for semantic search queries
- **Real-time Updates**: <60s for monitoring notifications
- **Scraping Jobs**: Status updates via WebSocket within seconds
- **Trend Calculation**: Background processing every few minutes for instant API responses

### Phased Development Strategy

#### Phase 1: Core Loop & Resilience (Months 1-3)
- **Focus**: Build scraping loop and Spinal Cord fallback
- **Components**: Dendrites, Neurons, Synaptic Vesicle, Spinal Cord
- **KPIs**: Scrape coverage >95%, recipe accuracy >99%

#### Phase 2: Cortex & Thalamus (Month 4)
- **Focus**: Develop hub and NLP engine
- **Components**: Central Cortex, Thalamus, basic dashboard
- **KPIs**: NLP accuracy >85%, system health monitoring

#### Phase 3: Go-to-Market (Month 5)
- **Focus**: Launch initial APIs with developer onboarding
- **Components**: Axon Interface, documentation, API sandbox
- **KPIs**: Developer signups, time-to-first-API-call <5 minutes

#### Phase 4: Product Expansion (Months 6-7)
- **Focus**: Add specialized APIs and integrations
- **Components**: All 12 API endpoints, webhook systems
- **KPIs**: API adoption rate, webhook delivery success >99%

#### Phase 5: Monetization (Month 8+)
- **Focus**: Introduce paid tiers and enterprise features
- **Components**: Billing system, advanced analytics
- **KPIs**: Free-to-paid conversion rate, monthly recurring revenue

This comprehensive design document provides the technical foundation for implementing Project Synapse, with detailed specifications for each component, clear data models, robust error handling, and a thorough testing strategy that covers all aspects of your original blueprint.