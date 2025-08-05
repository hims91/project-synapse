"""
Database performance optimization system for Project Synapse.

This module provides comprehensive database performance optimization including:
- Advanced connection pooling with monitoring
- Query performance analysis and optimization
- Automatic index recommendations and creation
- Database performance monitoring and alerting
"""

from .performance_optimizer import (
    QueryType,
    OptimizationLevel,
    QueryStats,
    IndexRecommendation,
    PerformanceConfig,
    ConnectionPoolManager,
    QueryOptimizer,
    IndexOptimizer,
    DatabasePerformanceOptimizer,
    get_performance_optimizer,
    initialize_performance_optimizer,
    shutdown_performance_optimizer
)

__all__ = [
    # Enums and data classes
    'QueryType',
    'OptimizationLevel',
    'QueryStats',
    'IndexRecommendation',
    'PerformanceConfig',
    
    # Core classes
    'ConnectionPoolManager',
    'QueryOptimizer',
    'IndexOptimizer',
    'DatabasePerformanceOptimizer',
    
    # Factory and lifecycle functions
    'get_performance_optimizer',
    'initialize_performance_optimizer',
    'shutdown_performance_optimizer'
]