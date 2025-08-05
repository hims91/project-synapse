"""
Database performance optimization system for Project Synapse.

Provides query optimization, connection pooling, index management,
and performance monitoring for PostgreSQL databases.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from collections import defaultdict, deque

import asyncpg
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select

from ..logging_config import get_logger
from ..metrics_collector import get_metrics_collector
from ..config import get_settings


class QueryType(str, Enum):
    """Query type enumeration."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    UNKNOWN = "unknown"


class OptimizationLevel(str, Enum):
    """Optimization level enumeration."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


@dataclass
class QueryStats:
    """Query execution statistics."""
    query_hash: str
    query_type: QueryType
    execution_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    last_executed: Optional[datetime] = None
    error_count: int = 0
    rows_affected: int = 0
    
    def update(self, duration: float, rows: int = 0, error: bool = False):
        """Update query statistics."""
        self.execution_count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.avg_duration = self.total_duration / self.execution_count
        self.last_executed = datetime.utcnow()
        self.rows_affected += rows
        
        if error:
            self.error_count += 1


@dataclass
class IndexRecommendation:
    """Database index recommendation."""
    table_name: str
    columns: List[str]
    index_type: str = "btree"
    reason: str = ""
    estimated_benefit: float = 0.0
    query_patterns: List[str] = field(default_factory=list)
    created: bool = False
    creation_time: Optional[datetime] = None


@dataclass
class PerformanceConfig:
    """Database performance configuration."""
    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Query optimization settings
    slow_query_threshold: float = 1.0  # seconds
    query_cache_size: int = 1000
    enable_query_logging: bool = True
    
    # Index optimization settings
    auto_create_indexes: bool = False
    index_usage_threshold: int = 100
    analyze_frequency: int = 3600  # seconds
    
    # Performance monitoring
    stats_retention_days: int = 7
    monitoring_interval: int = 60  # seconds
    
    # Optimization levels
    optimization_level: OptimizationLevel = OptimizationLevel.INTERMEDIATE


class ConnectionPoolManager:
    """Advanced connection pool manager."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger(__name__, 'connection_pool')
        self.metrics = get_metrics_collector()
        
        self.engine: Optional[Engine] = None
        self.pool_stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'connections_active': 0,
            'connections_idle': 0,
            'pool_overflows': 0,
            'connection_errors': 0
        }
    
    async def initialize(self, database_url: str) -> None:
        """Initialize the connection pool."""
        try:
            self.engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,
                echo=False,  # Set to True for SQL logging
                future=True
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get a database session from the pool."""
        if not self.engine:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            session = AsyncSession(self.engine, expire_on_commit=False)
            self.pool_stats['connections_active'] += 1
            return session
            
        except Exception as e:
            self.pool_stats['connection_errors'] += 1
            self.logger.error(f"Error getting database session: {e}")
            raise
    
    async def close_session(self, session: AsyncSession) -> None:
        """Close a database session."""
        try:
            await session.close()
            self.pool_stats['connections_active'] -= 1
            
        except Exception as e:
            self.logger.error(f"Error closing database session: {e}")
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        if not self.engine:
            return {'status': 'not_initialized'}
        
        pool = self.engine.pool
        
        return {
            'status': 'active',
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid(),
            'stats': self.pool_stats
        }
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.logger.info("Database connection pool closed")


class QueryOptimizer:
    """Query optimization and analysis system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger(__name__, 'query_optimizer')
        self.metrics = get_metrics_collector()
        
        self.query_stats: Dict[str, QueryStats] = {}
        self.slow_queries: deque = deque(maxlen=100)
        self.query_cache: Dict[str, Any] = {}
        
        # Query pattern analysis
        self.common_patterns = defaultdict(int)
        self.table_access_patterns = defaultdict(set)
    
    def _hash_query(self, query: str) -> str:
        """Generate a hash for query normalization."""
        # Normalize query by removing literals and whitespace
        normalized = re.sub(r'\b\d+\b', '?', query.lower())
        normalized = re.sub(r"'[^']*'", '?', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type."""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif query_lower.startswith('create'):
            return QueryType.CREATE
        elif query_lower.startswith('drop'):
            return QueryType.DROP
        elif query_lower.startswith('alter'):
            return QueryType.ALTER
        else:
            return QueryType.UNKNOWN
    
    def _extract_tables(self, query: str) -> Set[str]:
        """Extract table names from query."""
        # Simple regex-based table extraction
        # In production, you'd want a proper SQL parser
        tables = set()
        
        # Look for FROM and JOIN clauses
        from_pattern = r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_pattern = r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        update_pattern = r'\bupdate\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        insert_pattern = r'\binto\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for pattern in [from_pattern, join_pattern, update_pattern, insert_pattern]:
            matches = re.findall(pattern, query.lower())
            tables.update(matches)
        
        return tables
    
    async def analyze_query(self, query: str, duration: float, rows_affected: int = 0, error: bool = False) -> None:
        """Analyze query performance."""
        query_hash = self._hash_query(query)
        query_type = self._classify_query(query)
        
        # Update query statistics
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_type=query_type
            )
        
        self.query_stats[query_hash].update(duration, rows_affected, error)
        
        # Track slow queries
        if duration > self.config.slow_query_threshold:
            self.slow_queries.append({
                'query': query,
                'duration': duration,
                'timestamp': datetime.utcnow(),
                'rows_affected': rows_affected
            })
            
            self.logger.warning(f"Slow query detected: {duration:.2f}s - {query[:100]}...")
        
        # Analyze table access patterns
        tables = self._extract_tables(query)
        for table in tables:
            self.table_access_patterns[table].add(query_hash)
        
        # Record metrics
        counter = self.metrics.get_counter('database_queries_total')
        counter.increment(1, query_type=query_type.value, status='error' if error else 'success')
        
        histogram = self.metrics.get_histogram('database_query_duration_seconds')
        histogram.observe(duration, query_type=query_type.value)
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries."""
        sorted_stats = sorted(
            self.query_stats.values(),
            key=lambda s: s.avg_duration,
            reverse=True
        )
        
        return [
            {
                'query_hash': stat.query_hash,
                'query_type': stat.query_type.value,
                'avg_duration': stat.avg_duration,
                'execution_count': stat.execution_count,
                'total_duration': stat.total_duration,
                'error_rate': stat.error_count / stat.execution_count if stat.execution_count > 0 else 0
            }
            for stat in sorted_stats[:limit]
        ]
    
    def get_query_recommendations(self) -> List[str]:
        """Get query optimization recommendations."""
        recommendations = []
        
        # Analyze slow queries
        slow_queries = self.get_slow_queries(5)
        for query_info in slow_queries:
            if query_info['avg_duration'] > 2.0:
                recommendations.append(
                    f"Query {query_info['query_hash'][:8]} is very slow "
                    f"({query_info['avg_duration']:.2f}s avg). Consider adding indexes or rewriting."
                )
        
        # Analyze high-frequency queries
        high_freq_queries = sorted(
            self.query_stats.values(),
            key=lambda s: s.execution_count,
            reverse=True
        )[:5]
        
        for stat in high_freq_queries:
            if stat.execution_count > 1000 and stat.avg_duration > 0.5:
                recommendations.append(
                    f"High-frequency query {stat.query_hash[:8]} "
                    f"({stat.execution_count} executions) could benefit from optimization."
                )
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query optimizer statistics."""
        total_queries = sum(stat.execution_count for stat in self.query_stats.values())
        total_errors = sum(stat.error_count for stat in self.query_stats.values())
        
        query_type_counts = defaultdict(int)
        for stat in self.query_stats.values():
            query_type_counts[stat.query_type.value] += stat.execution_count
        
        return {
            'total_queries': total_queries,
            'unique_queries': len(self.query_stats),
            'total_errors': total_errors,
            'error_rate': total_errors / total_queries if total_queries > 0 else 0,
            'slow_queries_count': len(self.slow_queries),
            'query_types': dict(query_type_counts),
            'most_accessed_tables': dict(
                sorted(
                    {table: len(queries) for table, queries in self.table_access_patterns.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )
        }


class IndexOptimizer:
    """Database index optimization system."""
    
    def __init__(self, config: PerformanceConfig, pool_manager: ConnectionPoolManager):
        self.config = config
        self.pool_manager = pool_manager
        self.logger = get_logger(__name__, 'index_optimizer')
        self.metrics = get_metrics_collector()
        
        self.index_recommendations: List[IndexRecommendation] = []
        self.existing_indexes: Dict[str, List[str]] = {}
        self.index_usage_stats: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_indexes(self) -> None:
        """Analyze existing indexes and their usage."""
        try:
            session = await self.pool_manager.get_session()
            
            # Get existing indexes
            result = await session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """))
            
            indexes = result.fetchall()
            self.existing_indexes.clear()
            
            for row in indexes:
                table_name = row.tablename
                if table_name not in self.existing_indexes:
                    self.existing_indexes[table_name] = []
                self.existing_indexes[table_name].append({
                    'name': row.indexname,
                    'definition': row.indexdef
                })
            
            # Get index usage statistics
            result = await session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch,
                    idx_scan
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
            """))
            
            usage_stats = result.fetchall()
            self.index_usage_stats.clear()
            
            for row in usage_stats:
                key = f"{row.tablename}.{row.indexname}"
                self.index_usage_stats[key] = {
                    'tuples_read': row.idx_tup_read,
                    'tuples_fetched': row.idx_tup_fetch,
                    'scans': row.idx_scan
                }
            
            await self.pool_manager.close_session(session)
            
            self.logger.info(f"Analyzed {len(indexes)} indexes across {len(self.existing_indexes)} tables")
            
        except Exception as e:
            self.logger.error(f"Error analyzing indexes: {e}")
    
    async def get_table_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get table statistics for optimization analysis."""
        try:
            session = await self.pool_manager.get_session()
            
            result = await session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_live_tup,
                    n_dead_tup,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
            """))
            
            stats = {}
            for row in result.fetchall():
                stats[row.tablename] = {
                    'inserts': row.n_tup_ins,
                    'updates': row.n_tup_upd,
                    'deletes': row.n_tup_del,
                    'live_tuples': row.n_live_tup,
                    'dead_tuples': row.n_dead_tup,
                    'last_vacuum': row.last_vacuum,
                    'last_autovacuum': row.last_autovacuum,
                    'last_analyze': row.last_analyze,
                    'last_autoanalyze': row.last_autoanalyze
                }
            
            await self.pool_manager.close_session(session)
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting table stats: {e}")
            return {}
    
    def generate_index_recommendations(self, query_stats: Dict[str, QueryStats]) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns."""
        recommendations = []
        
        # Analyze slow SELECT queries for potential index opportunities
        slow_selects = [
            stat for stat in query_stats.values()
            if stat.query_type == QueryType.SELECT and stat.avg_duration > 1.0
        ]
        
        for stat in slow_selects:
            # This is a simplified example - in practice, you'd need
            # to parse the actual queries to identify WHERE clauses,
            # JOIN conditions, and ORDER BY clauses
            
            recommendation = IndexRecommendation(
                table_name="example_table",  # Would be extracted from query
                columns=["example_column"],  # Would be extracted from WHERE clauses
                reason=f"Slow query optimization (avg: {stat.avg_duration:.2f}s)",
                estimated_benefit=stat.avg_duration * 0.5,  # Estimated improvement
                query_patterns=[stat.query_hash]
            )
            recommendations.append(recommendation)
        
        # Remove duplicates and sort by estimated benefit
        unique_recommendations = {}
        for rec in recommendations:
            key = f"{rec.table_name}.{'.'.join(rec.columns)}"
            if key not in unique_recommendations or rec.estimated_benefit > unique_recommendations[key].estimated_benefit:
                unique_recommendations[key] = rec
        
        return sorted(unique_recommendations.values(), key=lambda r: r.estimated_benefit, reverse=True)
    
    async def create_recommended_indexes(self, recommendations: List[IndexRecommendation]) -> int:
        """Create recommended indexes if auto-creation is enabled."""
        if not self.config.auto_create_indexes:
            self.logger.info("Auto index creation is disabled")
            return 0
        
        created_count = 0
        
        try:
            session = await self.pool_manager.get_session()
            
            for rec in recommendations:
                if rec.created:
                    continue
                
                # Check if index already exists
                existing = self.existing_indexes.get(rec.table_name, [])
                if any(rec.columns[0] in idx['definition'] for idx in existing):
                    self.logger.info(f"Index on {rec.table_name}.{rec.columns[0]} already exists")
                    continue
                
                # Create index
                index_name = f"idx_{rec.table_name}_{'_'.join(rec.columns)}"
                columns_str = ', '.join(rec.columns)
                
                create_sql = f"""
                    CREATE INDEX CONCURRENTLY {index_name} 
                    ON {rec.table_name} USING {rec.index_type} ({columns_str})
                """
                
                try:
                    await session.execute(text(create_sql))
                    await session.commit()
                    
                    rec.created = True
                    rec.creation_time = datetime.utcnow()
                    created_count += 1
                    
                    self.logger.info(f"Created index: {index_name}")
                    
                except Exception as e:
                    await session.rollback()
                    self.logger.error(f"Failed to create index {index_name}: {e}")
            
            await self.pool_manager.close_session(session)
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
        
        return created_count
    
    def get_unused_indexes(self) -> List[str]:
        """Identify potentially unused indexes."""
        unused_indexes = []
        
        for index_key, stats in self.index_usage_stats.items():
            if stats['scans'] < self.config.index_usage_threshold:
                unused_indexes.append(index_key)
        
        return unused_indexes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index optimizer statistics."""
        total_indexes = sum(len(indexes) for indexes in self.existing_indexes.values())
        unused_indexes = len(self.get_unused_indexes())
        
        return {
            'total_indexes': total_indexes,
            'tables_with_indexes': len(self.existing_indexes),
            'unused_indexes': unused_indexes,
            'recommendations_generated': len(self.index_recommendations),
            'recommendations_created': sum(1 for rec in self.index_recommendations if rec.created)
        }


class DatabasePerformanceOptimizer:
    """Main database performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.logger = get_logger(__name__, 'db_optimizer')
        self.metrics = get_metrics_collector()
        
        self.pool_manager = ConnectionPoolManager(self.config)
        self.query_optimizer = QueryOptimizer(self.config)
        self.index_optimizer = IndexOptimizer(self.config, self.pool_manager)
        
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        self.stats = {
            'optimization_runs': 0,
            'indexes_created': 0,
            'queries_analyzed': 0,
            'recommendations_generated': 0
        }
    
    async def initialize(self, database_url: str) -> None:
        """Initialize the performance optimizer."""
        try:
            await self.pool_manager.initialize(database_url)
            await self.index_optimizer.analyze_indexes()
            
            self.logger.info("Database performance optimizer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance optimizer: {e}")
            raise
    
    async def analyze_query_performance(self, query: str, duration: float, rows_affected: int = 0, error: bool = False) -> None:
        """Analyze query performance (called by query execution wrapper)."""
        await self.query_optimizer.analyze_query(query, duration, rows_affected, error)
        self.stats['queries_analyzed'] += 1
    
    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete optimization cycle."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting database optimization cycle")
            
            # Analyze current indexes
            await self.index_optimizer.analyze_indexes()
            
            # Generate index recommendations
            recommendations = self.index_optimizer.generate_index_recommendations(
                self.query_optimizer.query_stats
            )
            
            self.index_optimizer.index_recommendations = recommendations
            self.stats['recommendations_generated'] += len(recommendations)
            
            # Create recommended indexes if enabled
            created_indexes = await self.index_optimizer.create_recommended_indexes(recommendations)
            self.stats['indexes_created'] += created_indexes
            
            # Get optimization results
            results = {
                'duration': time.time() - start_time,
                'recommendations_generated': len(recommendations),
                'indexes_created': created_indexes,
                'slow_queries': len(self.query_optimizer.get_slow_queries()),
                'query_recommendations': self.query_optimizer.get_query_recommendations(),
                'unused_indexes': self.index_optimizer.get_unused_indexes()
            }
            
            self.stats['optimization_runs'] += 1
            
            self.logger.info(
                f"Optimization cycle completed: {len(recommendations)} recommendations, "
                f"{created_indexes} indexes created in {results['duration']:.2f}s"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle: {e}")
            return {'error': str(e)}
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.running:
            self.logger.warning("Performance monitoring is already running")
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_worker())
        self.logger.info("Database performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.running:
            return
        
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        
        self.logger.info("Database performance monitoring stopped")
    
    async def _monitoring_worker(self) -> None:
        """Background worker for performance monitoring."""
        self.logger.info("Database performance monitoring worker started")
        
        while self.running:
            try:
                # Run optimization cycle periodically
                await self.run_optimization_cycle()
                
                # Update metrics
                pool_status = await self.pool_manager.get_pool_status()
                
                gauge = self.metrics.get_gauge('database_connections_active')
                gauge.set(pool_status.get('checked_out', 0))
                
                gauge = self.metrics.get_gauge('database_connections_idle')
                gauge.set(pool_status.get('checked_in', 0))
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
        
        self.logger.info("Database performance monitoring worker stopped")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        pool_status = await self.pool_manager.get_pool_status()
        table_stats = await self.index_optimizer.get_table_stats()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'connection_pool': pool_status,
            'query_optimizer': self.query_optimizer.get_stats(),
            'index_optimizer': self.index_optimizer.get_stats(),
            'table_stats': table_stats,
            'overall_stats': self.stats,
            'slow_queries': self.query_optimizer.get_slow_queries(10),
            'index_recommendations': [
                {
                    'table': rec.table_name,
                    'columns': rec.columns,
                    'reason': rec.reason,
                    'estimated_benefit': rec.estimated_benefit,
                    'created': rec.created
                }
                for rec in self.index_optimizer.index_recommendations[:10]
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the performance optimizer."""
        await self.stop_monitoring()
        await self.pool_manager.close()
        self.logger.info("Database performance optimizer shutdown completed")


# Global performance optimizer instance
_performance_optimizer: Optional[DatabasePerformanceOptimizer] = None


def get_performance_optimizer() -> DatabasePerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        settings = get_settings()
        config = PerformanceConfig(
            pool_size=getattr(settings, 'db_pool_size', 20),
            slow_query_threshold=getattr(settings, 'db_slow_query_threshold', 1.0),
            auto_create_indexes=getattr(settings, 'db_auto_create_indexes', False),
            optimization_level=OptimizationLevel(getattr(settings, 'db_optimization_level', 'intermediate'))
        )
        _performance_optimizer = DatabasePerformanceOptimizer(config)
    return _performance_optimizer


async def initialize_performance_optimizer(database_url: str) -> None:
    """Initialize the global performance optimizer."""
    optimizer = get_performance_optimizer()
    await optimizer.initialize(database_url)


async def shutdown_performance_optimizer() -> None:
    """Shutdown the global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer:
        await _performance_optimizer.shutdown()
        _performance_optimizer = None