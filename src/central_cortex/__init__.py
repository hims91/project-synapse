"""
Central Cortex - Hub Server
Layer 3: Cerebral Cortex

This module implements the main FastAPI hub server for Project Synapse.
The Central Cortex coordinates all system components and provides the main API.

The Central Cortex provides:
- FastAPI application with comprehensive middleware
- Authentication and API key management
- Content management and search endpoints
- System health monitoring and alerting
- Real-time WebSocket monitoring
- Rate limiting and security features
"""

from .app import app, create_app, run_server

__all__ = [
    'app',
    'create_app', 
    'run_server'
]