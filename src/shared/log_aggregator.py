"""
Log aggregation and centralized logging infrastructure.

Provides log collection, filtering, and forwarding to external systems
like ELK stack, Datadog, or other log management platforms.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import gzip
from pathlib import Path

from .logging_config import get_logger


class LogLevel(str, Enum):
    """Log levels for filtering."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogDestination(str, Enum):
    """Log destination types."""
    ELASTICSEARCH = "elasticsearch"
    DATADOG = "datadog"
    CLOUDWATCH = "cloudwatch"
    WEBHOOK = "webhook"
    FILE = "file"
    CONSOLE = "console"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger: str
    message: str
    correlation_id: str
    user_id: str
    request_id: str
    component: str
    operation: str
    module: str
    function: str
    line: int
    process_id: int
    thread_id: int
    exception: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class LogDestinationConfig:
    """Configuration for log destination."""
    type: LogDestination
    url: Optional[str] = None
    api_key: Optional[str] = None
    index: Optional[str] = None
    batch_size: int = 100
    flush_interval: int = 30
    compression: bool = True
    retry_attempts: int = 3
    retry_delay: int = 5
    filters: Optional[List[str]] = None
    min_level: LogLevel = LogLevel.INFO
    enabled: bool = True


class LogFilter:
    """Log filtering functionality."""
    
    def __init__(self):
        self.level_hierarchy = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
    
    def should_forward(
        self,
        log_entry: LogEntry,
        min_level: LogLevel,
        filters: Optional[List[str]] = None
    ) -> bool:
        """Check if log entry should be forwarded."""
        # Check log level
        entry_level = self.level_hierarchy.get(LogLevel(log_entry.level), 0)
        min_level_value = self.level_hierarchy.get(min_level, 0)
        
        if entry_level < min_level_value:
            return False
        
        # Check filters
        if filters:
            for filter_rule in filters:
                if not self._apply_filter(log_entry, filter_rule):
                    return False
        
        return True
    
    def _apply_filter(self, log_entry: LogEntry, filter_rule: str) -> bool:
        """Apply a single filter rule."""
        # Simple filter rules for now
        # Format: "field:value" or "field:!value" (negation)
        if ':' not in filter_rule:
            return True
        
        field, value = filter_rule.split(':', 1)
        negate = value.startswith('!')
        if negate:
            value = value[1:]
        
        entry_value = getattr(log_entry, field, '')
        match = value.lower() in str(entry_value).lower()
        
        return not match if negate else match


class LogBatch:
    """Batch of log entries for efficient forwarding."""
    
    def __init__(self, max_size: int = 100):
        self.entries: List[LogEntry] = []
        self.max_size = max_size
        self.created_at = time.time()
    
    def add(self, entry: LogEntry) -> bool:
        """Add entry to batch. Returns True if batch is full."""
        self.entries.append(entry)
        return len(self.entries) >= self.max_size
    
    def is_ready(self, flush_interval: int) -> bool:
        """Check if batch is ready for flushing."""
        return (
            len(self.entries) >= self.max_size or
            time.time() - self.created_at >= flush_interval
        )
    
    def to_json(self, compressed: bool = False) -> bytes:
        """Convert batch to JSON bytes."""
        data = [entry.to_dict() for entry in self.entries]
        json_data = json.dumps(data, default=str).encode('utf-8')
        
        if compressed:
            return gzip.compress(json_data)
        return json_data
    
    def clear(self):
        """Clear the batch."""
        self.entries.clear()
        self.created_at = time.time()


class LogForwarder:
    """Forwards logs to external destinations."""
    
    def __init__(self, config: LogDestinationConfig):
        self.config = config
        self.logger = get_logger(__name__, 'log_forwarder')
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {
            'logs_forwarded': 0,
            'logs_failed': 0,
            'batches_sent': 0,
            'last_forward_time': None,
            'errors': []
        }
    
    async def start(self):
        """Start the forwarder."""
        if self.config.type in [LogDestination.ELASTICSEARCH, LogDestination.WEBHOOK, LogDestination.DATADOG]:
            self.session = aiohttp.ClientSession()
        
        self.logger.info(
            f"Log forwarder started for {self.config.type}",
            operation="start_forwarder",
            destination=self.config.type.value,
            url=self.config.url
        )
    
    async def stop(self):
        """Stop the forwarder."""
        if self.session:
            await self.session.close()
        
        self.logger.info(
            f"Log forwarder stopped for {self.config.type}",
            operation="stop_forwarder",
            destination=self.config.type.value
        )
    
    async def forward_batch(self, batch: LogBatch) -> bool:
        """Forward a batch of logs."""
        if not self.config.enabled or not batch.entries:
            return True
        
        try:
            success = False
            
            if self.config.type == LogDestination.ELASTICSEARCH:
                success = await self._forward_to_elasticsearch(batch)
            elif self.config.type == LogDestination.DATADOG:
                success = await self._forward_to_datadog(batch)
            elif self.config.type == LogDestination.WEBHOOK:
                success = await self._forward_to_webhook(batch)
            elif self.config.type == LogDestination.FILE:
                success = await self._forward_to_file(batch)
            elif self.config.type == LogDestination.CONSOLE:
                success = await self._forward_to_console(batch)
            
            if success:
                self.stats['logs_forwarded'] += len(batch.entries)
                self.stats['batches_sent'] += 1
                self.stats['last_forward_time'] = datetime.utcnow().isoformat()
            else:
                self.stats['logs_failed'] += len(batch.entries)
            
            return success
        
        except Exception as e:
            self.logger.error(
                f"Error forwarding logs to {self.config.type}",
                operation="forward_batch",
                error=str(e),
                batch_size=len(batch.entries)
            )
            self.stats['logs_failed'] += len(batch.entries)
            self.stats['errors'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'batch_size': len(batch.entries)
            })
            return False
    
    async def _forward_to_elasticsearch(self, batch: LogBatch) -> bool:
        """Forward logs to Elasticsearch."""
        if not self.session or not self.config.url:
            return False
        
        # Prepare bulk index request
        bulk_data = []
        for entry in batch.entries:
            index_action = {
                "index": {
                    "_index": self.config.index or "synapse-logs",
                    "_type": "_doc"
                }
            }
            bulk_data.append(json.dumps(index_action))
            bulk_data.append(entry.to_json())
        
        bulk_body = '\n'.join(bulk_data) + '\n'
        
        headers = {'Content-Type': 'application/x-ndjson'}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        async with self.session.post(
            f"{self.config.url}/_bulk",
            data=bulk_body,
            headers=headers
        ) as response:
            return response.status == 200
    
    async def _forward_to_datadog(self, batch: LogBatch) -> bool:
        """Forward logs to Datadog."""
        if not self.session or not self.config.api_key:
            return False
        
        logs_data = [entry.to_dict() for entry in batch.entries]
        
        headers = {
            'Content-Type': 'application/json',
            'DD-API-KEY': self.config.api_key
        }
        
        async with self.session.post(
            'https://http-intake.logs.datadoghq.com/v1/input',
            json=logs_data,
            headers=headers
        ) as response:
            return response.status == 200
    
    async def _forward_to_webhook(self, batch: LogBatch) -> bool:
        """Forward logs to webhook."""
        if not self.session or not self.config.url:
            return False
        
        data = batch.to_json(compressed=self.config.compression)
        headers = {'Content-Type': 'application/json'}
        
        if self.config.compression:
            headers['Content-Encoding'] = 'gzip'
        
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        async with self.session.post(
            self.config.url,
            data=data,
            headers=headers
        ) as response:
            return 200 <= response.status < 300
    
    async def _forward_to_file(self, batch: LogBatch) -> bool:
        """Forward logs to file."""
        try:
            log_file = Path(self.config.url or 'logs/aggregated.log')
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'a') as f:
                for entry in batch.entries:
                    f.write(entry.to_json() + '\n')
            
            return True
        except Exception:
            return False
    
    async def _forward_to_console(self, batch: LogBatch) -> bool:
        """Forward logs to console."""
        try:
            for entry in batch.entries:
                print(entry.to_json())
            return True
        except Exception:
            return False


class LogAggregator:
    """Central log aggregation service."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'log_aggregator')
        self.destinations: Dict[str, LogForwarder] = {}
        self.batches: Dict[str, LogBatch] = {}
        self.filter = LogFilter()
        self.running = False
        self.flush_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'logs_received': 0,
            'logs_filtered': 0,
            'logs_forwarded': 0,
            'start_time': None,
            'destinations_active': 0
        }
    
    def add_destination(self, name: str, config: LogDestinationConfig):
        """Add a log destination."""
        forwarder = LogForwarder(config)
        self.destinations[name] = forwarder
        self.batches[name] = LogBatch(config.batch_size)
        
        self.logger.info(
            f"Added log destination: {name}",
            operation="add_destination",
            destination_type=config.type.value,
            destination_name=name
        )
    
    def remove_destination(self, name: str):
        """Remove a log destination."""
        if name in self.destinations:
            del self.destinations[name]
            del self.batches[name]
            
            self.logger.info(
                f"Removed log destination: {name}",
                operation="remove_destination",
                destination_name=name
            )
    
    async def start(self):
        """Start the log aggregator."""
        if self.running:
            return
        
        self.running = True
        self.stats['start_time'] = datetime.utcnow().isoformat()
        
        # Start all forwarders
        for forwarder in self.destinations.values():
            await forwarder.start()
        
        # Start flush task
        self.flush_task = asyncio.create_task(self._flush_loop())
        
        self.stats['destinations_active'] = len(self.destinations)
        
        self.logger.info(
            "Log aggregator started",
            operation="start_aggregator",
            destinations_count=len(self.destinations)
        )
    
    async def stop(self):
        """Stop the log aggregator."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel flush task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining logs
        await self._flush_all_batches()
        
        # Stop all forwarders
        for forwarder in self.destinations.values():
            await forwarder.stop()
        
        self.logger.info(
            "Log aggregator stopped",
            operation="stop_aggregator"
        )
    
    async def process_log(self, log_data: Dict[str, Any]):
        """Process a log entry."""
        try:
            # Convert to LogEntry
            log_entry = LogEntry(
                timestamp=log_data.get('timestamp', datetime.utcnow().isoformat()),
                level=log_data.get('level', 'INFO'),
                logger=log_data.get('logger', 'unknown'),
                message=log_data.get('message', ''),
                correlation_id=log_data.get('correlation_id', 'unknown'),
                user_id=log_data.get('user_id', 'anonymous'),
                request_id=log_data.get('request_id', 'no-request'),
                component=log_data.get('component', 'unknown'),
                operation=log_data.get('operation', 'unknown'),
                module=log_data.get('module', 'unknown'),
                function=log_data.get('function', 'unknown'),
                line=log_data.get('line', 0),
                process_id=log_data.get('process_id', 0),
                thread_id=log_data.get('thread_id', 0),
                exception=log_data.get('exception'),
                extra={k: v for k, v in log_data.items() if k not in [
                    'timestamp', 'level', 'logger', 'message', 'correlation_id',
                    'user_id', 'request_id', 'component', 'operation', 'module',
                    'function', 'line', 'process_id', 'thread_id', 'exception'
                ]}
            )
            
            self.stats['logs_received'] += 1
            
            # Process for each destination
            for name, forwarder in self.destinations.items():
                config = forwarder.config
                
                # Apply filters
                if self.filter.should_forward(log_entry, config.min_level, config.filters):
                    batch = self.batches[name]
                    is_full = batch.add(log_entry)
                    
                    # Flush if batch is full
                    if is_full:
                        await self._flush_batch(name, forwarder, batch)
                else:
                    self.stats['logs_filtered'] += 1
        
        except Exception as e:
            self.logger.error(
                "Error processing log entry",
                operation="process_log",
                error=str(e)
            )
    
    async def _flush_loop(self):
        """Background task to flush batches periodically."""
        while self.running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._flush_ready_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in flush loop",
                    operation="flush_loop",
                    error=str(e)
                )
    
    async def _flush_ready_batches(self):
        """Flush batches that are ready."""
        for name, batch in self.batches.items():
            forwarder = self.destinations[name]
            config = forwarder.config
            
            if batch.is_ready(config.flush_interval):
                await self._flush_batch(name, forwarder, batch)
    
    async def _flush_batch(self, name: str, forwarder: LogForwarder, batch: LogBatch):
        """Flush a specific batch."""
        if not batch.entries:
            return
        
        success = await forwarder.forward_batch(batch)
        
        if success:
            self.stats['logs_forwarded'] += len(batch.entries)
        
        batch.clear()
    
    async def _flush_all_batches(self):
        """Flush all remaining batches."""
        for name, batch in self.batches.items():
            if batch.entries:
                forwarder = self.destinations[name]
                await self._flush_batch(name, forwarder, batch)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        destination_stats = {}
        for name, forwarder in self.destinations.items():
            destination_stats[name] = forwarder.stats
        
        return {
            **self.stats,
            'destinations': destination_stats,
            'uptime_seconds': (
                (datetime.utcnow() - datetime.fromisoformat(self.stats['start_time'])).total_seconds()
                if self.stats['start_time'] else 0
            )
        }


# Global aggregator instance
_log_aggregator: Optional[LogAggregator] = None


def get_log_aggregator() -> LogAggregator:
    """Get the global log aggregator instance."""
    global _log_aggregator
    if _log_aggregator is None:
        _log_aggregator = LogAggregator()
    return _log_aggregator


async def setup_log_aggregation(destinations: Dict[str, LogDestinationConfig]):
    """Setup log aggregation with destinations."""
    aggregator = get_log_aggregator()
    
    for name, config in destinations.items():
        aggregator.add_destination(name, config)
    
    await aggregator.start()
    return aggregator