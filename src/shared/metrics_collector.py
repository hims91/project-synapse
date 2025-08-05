"""
Metrics collection system for Project Synapse.

Provides application metrics, performance monitoring, business KPIs,
and infrastructure monitoring capabilities.
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from .logging_config import get_logger


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SET = "set"


class MetricUnit(str, Enum):
    """Metric units."""
    COUNT = "count"
    BYTES = "bytes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    PERCENT = "percent"
    REQUESTS_PER_SECOND = "requests_per_second"
    ERRORS_PER_SECOND = "errors_per_second"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    unit: MetricUnit
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'unit': self.unit.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'labels': self.labels
        }


@dataclass
class MetricSeries:
    """A time series of metric values."""
    name: str
    metric_type: MetricType
    unit: MetricUnit
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_value(self, value: Union[int, float], timestamp: Optional[datetime] = None, **labels):
        """Add a value to the series."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        metric_value = MetricValue(
            name=self.name,
            value=value,
            metric_type=self.metric_type,
            unit=self.unit,
            timestamp=timestamp,
            tags=self.tags,
            labels=labels
        )
        
        self.values.append(metric_value)
        self.last_updated = timestamp
    
    def get_latest_value(self) -> Optional[MetricValue]:
        """Get the latest value."""
        return self.values[-1] if self.values else None
    
    def get_values_in_range(self, start_time: datetime, end_time: datetime) -> List[MetricValue]:
        """Get values within a time range."""
        return [
            value for value in self.values
            if start_time <= value.timestamp <= end_time
        ]
    
    def calculate_statistics(self, window_minutes: int = 5) -> Dict[str, float]:
        """Calculate statistics for recent values."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_values = [
            value.value for value in self.values
            if value.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'sum': sum(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        }


class Counter:
    """Counter metric that only increases."""
    
    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self, amount: Union[int, float] = 1, **labels):
        """Increment the counter."""
        with self._lock:
            self._value += amount
        
        # Record metric
        MetricsCollector.get_instance().record_metric(
            self.name, self._value, MetricType.COUNTER, MetricUnit.COUNT,
            tags=self.tags, **labels
        )
    
    def get_value(self) -> Union[int, float]:
        """Get current value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """Gauge metric that can increase or decrease."""
    
    def __init__(self, name: str, description: str = "", unit: MetricUnit = MetricUnit.COUNT, tags: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.tags = tags or {}
        self._value = 0
        self._lock = threading.Lock()
    
    def set(self, value: Union[int, float], **labels):
        """Set the gauge value."""
        with self._lock:
            self._value = value
        
        # Record metric
        MetricsCollector.get_instance().record_metric(
            self.name, self._value, MetricType.GAUGE, self.unit,
            tags=self.tags, **labels
        )
    
    def increment(self, amount: Union[int, float] = 1, **labels):
        """Increment the gauge."""
        with self._lock:
            self._value += amount
        
        # Record metric
        MetricsCollector.get_instance().record_metric(
            self.name, self._value, MetricType.GAUGE, self.unit,
            tags=self.tags, **labels
        )
    
    def decrement(self, amount: Union[int, float] = 1, **labels):
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount
        
        # Record metric
        MetricsCollector.get_instance().record_metric(
            self.name, self._value, MetricType.GAUGE, self.unit,
            tags=self.tags, **labels
        )
    
    def get_value(self) -> Union[int, float]:
        """Get current value."""
        with self._lock:
            return self._value


class Histogram:
    """Histogram metric for tracking distributions."""
    
    def __init__(self, name: str, description: str = "", unit: MetricUnit = MetricUnit.COUNT, 
                 buckets: List[float] = None, tags: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.tags = tags or {}
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self._bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._sum = 0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: Union[int, float], **labels):
        """Observe a value."""
        with self._lock:
            self._sum += value
            self._count += 1
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
        
        # Record metric
        MetricsCollector.get_instance().record_metric(
            self.name, value, MetricType.HISTOGRAM, self.unit,
            tags=self.tags, **labels
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            return {
                'count': self._count,
                'sum': self._sum,
                'mean': self._sum / self._count if self._count > 0 else 0,
                'buckets': self._bucket_counts.copy()
            }


class Timer:
    """Timer metric for measuring durations."""
    
    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.histogram = Histogram(f"{name}_duration", description, MetricUnit.SECONDS, tags=tags)
    
    def time(self, **labels):
        """Context manager for timing operations."""
        return TimerContext(self, labels)
    
    def record(self, duration: float, **labels):
        """Record a duration."""
        self.histogram.observe(duration, **labels)


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, timer: Timer, labels: Dict[str, str]):
        self.timer = timer
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.timer.record(duration, **self.labels)


class MetricsCollector:
    """Central metrics collection system."""
    
    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.logger = get_logger(__name__, 'metrics_collector')
        self.metrics: Dict[str, MetricSeries] = {}
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.timers: Dict[str, Timer] = {}
        
        # System metrics
        self.system_metrics_enabled = True
        self.system_metrics_interval = 30  # seconds
        self.system_metrics_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.request_counter = Counter('http_requests_total', 'Total HTTP requests')
        self.request_duration = Timer('http_request_duration', 'HTTP request duration')
        self.error_counter = Counter('errors_total', 'Total errors')
        
        # Business metrics
        self.business_metrics = {
            'jobs_created': Counter('jobs_created_total', 'Total jobs created'),
            'jobs_completed': Counter('jobs_completed_total', 'Total jobs completed'),
            'jobs_failed': Counter('jobs_failed_total', 'Total jobs failed'),
            'articles_scraped': Counter('articles_scraped_total', 'Total articles scraped'),
            'api_calls': Counter('api_calls_total', 'Total API calls'),
            'active_users': Gauge('active_users', 'Number of active users'),
            'queue_size': Gauge('queue_size', 'Current queue size'),
            'cache_hit_rate': Gauge('cache_hit_rate', 'Cache hit rate', MetricUnit.PERCENT),
        }
        
        # Statistics
        self.stats = {
            'metrics_recorded': 0,
            'start_time': datetime.utcnow(),
            'last_collection_time': None
        }
    
    @classmethod
    def get_instance(cls) -> 'MetricsCollector':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        unit: MetricUnit,
        timestamp: Optional[datetime] = None,
        tags: Dict[str, str] = None,
        **labels
    ):
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Get or create metric series
        series_key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        
        if series_key not in self.metrics:
            self.metrics[series_key] = MetricSeries(
                name=name,
                metric_type=metric_type,
                unit=unit,
                tags=tags or {}
            )
        
        # Add value to series
        self.metrics[series_key].add_value(value, timestamp, **labels)
        
        self.stats['metrics_recorded'] += 1
        self.stats['last_collection_time'] = timestamp
    
    def get_counter(self, name: str, description: str = "", tags: Dict[str, str] = None) -> Counter:
        """Get or create a counter."""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.counters:
            self.counters[key] = Counter(name, description, tags)
        return self.counters[key]
    
    def get_gauge(self, name: str, description: str = "", unit: MetricUnit = MetricUnit.COUNT, 
                  tags: Dict[str, str] = None) -> Gauge:
        """Get or create a gauge."""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.gauges:
            self.gauges[key] = Gauge(name, description, unit, tags)
        return self.gauges[key]
    
    def get_histogram(self, name: str, description: str = "", unit: MetricUnit = MetricUnit.COUNT,
                     buckets: List[float] = None, tags: Dict[str, str] = None) -> Histogram:
        """Get or create a histogram."""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.histograms:
            self.histograms[key] = Histogram(name, description, unit, buckets, tags)
        return self.histograms[key]
    
    def get_timer(self, name: str, description: str = "", tags: Dict[str, str] = None) -> Timer:
        """Get or create a timer."""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.timers:
            self.timers[key] = Timer(name, description, tags)
        return self.timers[key]
    
    async def start_system_metrics_collection(self):
        """Start collecting system metrics."""
        if not self.system_metrics_enabled or self.system_metrics_task:
            return
        
        self.system_metrics_task = asyncio.create_task(self._collect_system_metrics())
        self.logger.info("Started system metrics collection", operation="start_system_metrics")
    
    async def stop_system_metrics_collection(self):
        """Stop collecting system metrics."""
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
            try:
                await self.system_metrics_task
            except asyncio.CancelledError:
                pass
            self.system_metrics_task = None
        
        self.logger.info("Stopped system metrics collection", operation="stop_system_metrics")
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics."""
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric(
                    'system_cpu_percent', cpu_percent, MetricType.GAUGE, MetricUnit.PERCENT
                )
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_metric(
                    'system_memory_percent', memory.percent, MetricType.GAUGE, MetricUnit.PERCENT
                )
                self.record_metric(
                    'system_memory_used_bytes', memory.used, MetricType.GAUGE, MetricUnit.BYTES
                )
                self.record_metric(
                    'system_memory_available_bytes', memory.available, MetricType.GAUGE, MetricUnit.BYTES
                )
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_metric(
                    'system_disk_percent', (disk.used / disk.total) * 100, MetricType.GAUGE, MetricUnit.PERCENT
                )
                self.record_metric(
                    'system_disk_used_bytes', disk.used, MetricType.GAUGE, MetricUnit.BYTES
                )
                self.record_metric(
                    'system_disk_free_bytes', disk.free, MetricType.GAUGE, MetricUnit.BYTES
                )
                
                # Network metrics
                network = psutil.net_io_counters()
                self.record_metric(
                    'system_network_bytes_sent', network.bytes_sent, MetricType.COUNTER, MetricUnit.BYTES
                )
                self.record_metric(
                    'system_network_bytes_recv', network.bytes_recv, MetricType.COUNTER, MetricUnit.BYTES
                )
                
                # Process metrics
                process = psutil.Process()
                self.record_metric(
                    'process_cpu_percent', process.cpu_percent(), MetricType.GAUGE, MetricUnit.PERCENT
                )
                self.record_metric(
                    'process_memory_rss_bytes', process.memory_info().rss, MetricType.GAUGE, MetricUnit.BYTES
                )
                self.record_metric(
                    'process_open_files', process.num_fds() if hasattr(process, 'num_fds') else 0, 
                    MetricType.GAUGE, MetricUnit.COUNT
                )
                
                await asyncio.sleep(self.system_metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error collecting system metrics: {e}",
                    operation="collect_system_metrics"
                )
                await asyncio.sleep(self.system_metrics_interval)
    
    def get_metric_series(self, name: str, tags: Dict[str, str] = None) -> Optional[MetricSeries]:
        """Get a metric series."""
        series_key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        return self.metrics.get(series_key)
    
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all metric series."""
        return self.metrics.copy()
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            'total_metrics': len(self.metrics),
            'collection_stats': self.stats.copy(),
            'metrics': {}
        }
        
        for series_key, series in self.metrics.items():
            latest_value = series.get_latest_value()
            stats = series.calculate_statistics(window_minutes)
            
            summary['metrics'][series_key] = {
                'name': series.name,
                'type': series.metric_type.value,
                'unit': series.unit.value,
                'tags': series.tags,
                'latest_value': latest_value.value if latest_value else None,
                'latest_timestamp': latest_value.timestamp.isoformat() if latest_value else None,
                'statistics': stats,
                'value_count': len(series.values)
            }
        
        return summary
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format."""
        if format_type == 'json':
            return json.dumps(self.get_metrics_summary(), default=str, indent=2)
        elif format_type == 'prometheus':
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for series_key, series in self.metrics.items():
            latest_value = series.get_latest_value()
            if not latest_value:
                continue
            
            # Metric name
            metric_name = series.name.replace('-', '_').replace('.', '_')
            
            # Help text
            lines.append(f"# HELP {metric_name} {series.name}")
            lines.append(f"# TYPE {metric_name} {series.metric_type.value}")
            
            # Labels
            labels = []
            if series.tags:
                labels.extend([f'{k}="{v}"' for k, v in series.tags.items()])
            if latest_value.labels:
                labels.extend([f'{k}="{v}"' for k, v in latest_value.labels.items()])
            
            label_str = '{' + ','.join(labels) + '}' if labels else ''
            
            # Value
            lines.append(f"{metric_name}{label_str} {latest_value.value}")
        
        return '\n'.join(lines)
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.request_counter.increment(method=method, endpoint=endpoint, status=str(status_code))
        self.request_duration.record(duration, method=method, endpoint=endpoint)
        
        if status_code >= 400:
            self.error_counter.increment(method=method, endpoint=endpoint, status=str(status_code))
    
    def record_job_created(self, job_type: str, priority: str = 'normal'):
        """Record job creation."""
        self.business_metrics['jobs_created'].increment(job_type=job_type, priority=priority)
    
    def record_job_completed(self, job_type: str, duration: float, success: bool = True):
        """Record job completion."""
        if success:
            self.business_metrics['jobs_completed'].increment(job_type=job_type)
        else:
            self.business_metrics['jobs_failed'].increment(job_type=job_type)
        
        # Record duration
        job_timer = self.get_timer('job_duration', 'Job execution duration')
        job_timer.record(duration, job_type=job_type, success=str(success))
    
    def record_article_scraped(self, source: str, success: bool = True):
        """Record article scraping."""
        self.business_metrics['articles_scraped'].increment(source=source, success=str(success))
    
    def record_api_call(self, endpoint: str, user_tier: str = 'free'):
        """Record API call."""
        self.business_metrics['api_calls'].increment(endpoint=endpoint, tier=user_tier)
    
    def set_active_users(self, count: int):
        """Set active users count."""
        self.business_metrics['active_users'].set(count)
    
    def set_queue_size(self, queue_name: str, size: int):
        """Set queue size."""
        self.business_metrics['queue_size'].set(size, queue=queue_name)
    
    def set_cache_hit_rate(self, cache_name: str, hit_rate: float):
        """Set cache hit rate."""
        self.business_metrics['cache_hit_rate'].set(hit_rate * 100, cache=cache_name)


# Global metrics collector instance
def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return MetricsCollector.get_instance()


# Convenience functions
def increment_counter(name: str, amount: Union[int, float] = 1, **labels):
    """Increment a counter metric."""
    collector = get_metrics_collector()
    counter = collector.get_counter(name)
    counter.increment(amount, **labels)


def set_gauge(name: str, value: Union[int, float], unit: MetricUnit = MetricUnit.COUNT, **labels):
    """Set a gauge metric."""
    collector = get_metrics_collector()
    gauge = collector.get_gauge(name, unit=unit)
    gauge.set(value, **labels)


def observe_histogram(name: str, value: Union[int, float], unit: MetricUnit = MetricUnit.COUNT, **labels):
    """Observe a histogram metric."""
    collector = get_metrics_collector()
    histogram = collector.get_histogram(name, unit=unit)
    histogram.observe(value, **labels)


def time_operation(name: str, **labels):
    """Time an operation."""
    collector = get_metrics_collector()
    timer = collector.get_timer(name)
    return timer.time(**labels)