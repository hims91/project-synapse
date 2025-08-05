# Project Synapse - Neurons (Lightweight Scrapers) Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (minimal set for scrapers)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    httpx==0.25.2 \
    beautifulsoup4==4.12.2 \
    lxml==4.9.3 \
    redis==5.0.1 \
    websockets==12.0 \
    python-dotenv==1.0.0

# Copy only necessary source code
COPY src/neurons/ ./src/neurons/
COPY src/shared/ ./src/shared/
COPY src/dendrites/feed_parser.py ./src/dendrites/
COPY src/dendrites/url_resolver.py ./src/dendrites/

# Create cache directory
RUN mkdir -p cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV COMPONENT_TYPE=neurons

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the neuron scraper service
CMD ["python", "-m", "uvicorn", "src.neurons.main:app", "--host", "0.0.0.0", "--port", "8080"]