# Project Synapse: The Definitive Blueprint v2.2

**Motto:** "Feel the web. Think in data. Act with insight."

## 🌍 Overview

Project Synapse is the world's first self-learning, zero-cost intelligence network that proactively perceives the web in real-time, understands information at scale, and delivers actionable insights instantly. By leveraging free-tier cloud services and a brain-inspired design, it democratizes access to web intelligence.

## 🧠 Architecture

Project Synapse follows a brain-inspired, multi-layer architecture with components named after parts of the nervous system:

### Layer 0: Sensory Input Layer
- **Dendrites** - High-frequency RSS/Atom feed pollers for content discovery

### Layer 1: Perception Layer  
- **Neurons** - Fast, low-cost scrapers using pre-compiled recipes
- **Sensory Neurons** - Browser-based learners for new/changed/blocked domains

### Layer 2: Signal Network
- **Synaptic Vesicle** - PostgreSQL queue, recipe memory, and feed list
- **Signal Relay** - Dispatches tasks to Sensory Neurons
- **Spinal Cord** - Secondary, decentralized message bus for resilience

### Layer 3: Cerebral Cortex
- **Central Cortex** - FastAPI app, dashboard, and API gateway
- **Thalamus** - Text understanding, entity recognition, sentiment analysis

### Layer 4: Public Interface
- **Axon Interface** - API suite for developers and customers

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project-synapse
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Dashboard: http://localhost:3000
   - API Docs: http://localhost:8000/docs

### Manual Setup (Alternative)

1. **Python Backend**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

2. **Frontend Dashboard**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 📊 API Endpoints

### Core APIs
- **Content API** (`/content/*`) - Access to scraped articles
- **Semantic Search** (`/search`) - Natural language search
- **ScrapeDrop** (`/scrape/*`) - On-demand URL scraping
- **WebWatch** (`/monitoring/*`) - Keyword monitoring with webhooks

### Specialized APIs
- **FinMind** (`/financial/*`) - Financial news with sentiment analysis
- **Digestify** (`/summarize`) - Article summarization
- **InsightGraph** (`/relationships`) - Entity relationship extraction
- **MetaLens** (`/meta`) - Technical webpage intelligence
- **Chrono-Track** (`/tracking/*`) - Webpage change monitoring
- **Trends** (`/trends`) - Real-time trending topics
- **Headlines** (`/headlines`) - Curated top headlines
- **Bias Analysis** (`/analysis/narrative`) - Media bias analysis

## 🛠️ Development

### Project Structure
```
project-synapse/
├── src/
│   ├── dendrites/          # Feed polling system
│   ├── neurons/            # Lightweight scrapers
│   ├── sensory_neurons/    # Learning scrapers
│   ├── synaptic_vesicle/   # Database layer
│   ├── signal_relay/       # Task dispatcher
│   ├── spinal_cord/        # Fallback system
│   ├── central_cortex/     # Hub server
│   ├── thalamus/           # NLP engine
│   ├── axon_interface/     # Public APIs
│   └── shared/             # Common utilities
├── frontend/               # React dashboard
├── tests/                  # Test suite
├── docs/                   # Documentation
└── deployment/             # Deployment configs
```

### Running Tests
```bash
# Backend tests
pytest

# Frontend tests
cd frontend && npm test

# Integration tests
pytest tests/integration/

# Load tests
pytest tests/performance/
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Lint
flake8 src/
mypy src/

# Pre-commit hooks
pre-commit install
```

## 🚀 Deployment

### One-Click Deployment
```bash
# Deploy to staging
./deployment/scripts/deploy-staging.sh

# Deploy to production
./deployment/scripts/deploy-production.sh
```

### Manual Deployment
See [deployment documentation](docs/deployment.md) for detailed instructions.

## 📈 Monitoring

- **Health Check**: `/health`
- **Metrics**: `/metrics` (Prometheus format)
- **System Status**: Dashboard at `/dashboard`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Discussions: GitHub Discussions

---

**Project Synapse** - Democratizing web intelligence through brain-inspired architecture.