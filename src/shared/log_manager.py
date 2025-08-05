"""
Log level management and filtering system.

Provides dynamic log level management, filtering rules, and log analysis
capabilities for Project Synapse.
"""

import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Pattern
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import asyncio
from pathlib import Path

from .logging_config import get_logger, LogLevel


class FilterAction(str, Enum):
    """Actions for log filtering."""
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    ROUTE = "route"


class FilterOperator(str, Enum):
    """Operators for filter conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"


@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False
    
    def matches(self, log_data: Dict[str, Any]) -> bool:
        """Check if log data matches this condition."""
        field_value = self._get_field_value(log_data, self.field)
        
        if field_value is None:
            return False
        
        # Convert to string for string operations
        if isinstance(field_value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            if isinstance(self.value, str):
                compare_value = self.value.lower()
            else:
                compare_value = self.value
        else:
            compare_value = self.value
        
        # Apply operator
        if self.operator == FilterOperator.EQUALS:
            return field_value == compare_value
        elif self.operator == FilterOperator.NOT_EQUALS:
            return field_value != compare_value
        elif self.operator == FilterOperator.CONTAINS:
            return str(compare_value) in str(field_value)
        elif self.operator == FilterOperator.NOT_CONTAINS:
            return str(compare_value) not in str(field_value)
        elif self.operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(compare_value))
        elif self.operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(compare_value))
        elif self.operator == FilterOperator.REGEX:
            pattern = re.compile(str(compare_value), re.IGNORECASE if not self.case_sensitive else 0)
            return bool(pattern.search(str(field_value)))
        elif self.operator == FilterOperator.IN:
            return field_value in (compare_value if isinstance(compare_value, (list, set, tuple)) else [compare_value])
        elif self.operator == FilterOperator.NOT_IN:
            return field_value not in (compare_value if isinstance(compare_value, (list, set, tuple)) else [compare_value])
        elif self.operator == FilterOperator.GREATER_THAN:
            try:
                return float(field_value) > float(compare_value)
            except (ValueError, TypeError):
                return False
        elif self.operator == FilterOperator.LESS_THAN:
            try:
                return float(field_value) < float(compare_value)
            except (ValueError, TypeError):
                return False
        
        return False
    
    def _get_field_value(self, log_data: Dict[str, Any], field_path: str) -> Any:
        """Get field value from log data, supporting nested paths."""
        parts = field_path.split('.')
        value = log_data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value


@dataclass
class LogFilter:
    """A log filter with conditions and actions."""
    name: str
    description: str
    conditions: List[FilterCondition]
    action: FilterAction
    action_params: Dict[str, Any]
    enabled: bool = True
    priority: int = 0
    match_all: bool = True  # True = AND, False = OR
    
    def matches(self, log_data: Dict[str, Any]) -> bool:
        """Check if log data matches this filter."""
        if not self.enabled or not self.conditions:
            return False
        
        if self.match_all:
            # All conditions must match (AND)
            return all(condition.matches(log_data) for condition in self.conditions)
        else:
            # Any condition can match (OR)
            return any(condition.matches(log_data) for condition in self.conditions)
    
    def apply(self, log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply filter action to log data."""
        if not self.matches(log_data):
            return log_data
        
        if self.action == FilterAction.DENY:
            return None  # Filter out the log
        elif self.action == FilterAction.ALLOW:
            return log_data  # Pass through unchanged
        elif self.action == FilterAction.MODIFY:
            return self._apply_modifications(log_data)
        elif self.action == FilterAction.ROUTE:
            # Add routing information
            modified_data = log_data.copy()
            modified_data['_routing'] = self.action_params.get('destination', 'default')
            return modified_data
        
        return log_data
    
    def _apply_modifications(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to log data."""
        modified_data = log_data.copy()
        
        modifications = self.action_params.get('modifications', {})
        for field, value in modifications.items():
            if field.startswith('_add_'):
                # Add new field
                new_field = field[5:]  # Remove '_add_' prefix
                modified_data[new_field] = value
            elif field.startswith('_remove_'):
                # Remove field
                remove_field = field[8:]  # Remove '_remove_' prefix
                modified_data.pop(remove_field, None)
            else:
                # Modify existing field
                modified_data[field] = value
        
        return modified_data


class LogLevelManager:
    """Manages dynamic log levels for different components."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'log_level_manager')
        self.component_levels: Dict[str, str] = {}
        self.default_level = 'INFO'
        self.level_hierarchy = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
    
    def set_level(self, component: str, level: str):
        """Set log level for a component."""
        if level.upper() not in self.level_hierarchy:
            raise ValueError(f"Invalid log level: {level}")
        
        self.component_levels[component] = level.upper()
        
        # Apply to actual logger
        logger = logging.getLogger(component)
        logger.setLevel(getattr(logging, level.upper()))
        
        self.logger.info(
            f"Set log level for {component} to {level}",
            operation="set_log_level",
            component=component,
            level=level
        )
    
    def get_level(self, component: str) -> str:
        """Get log level for a component."""
        return self.component_levels.get(component, self.default_level)
    
    def set_default_level(self, level: str):
        """Set default log level."""
        if level.upper() not in self.level_hierarchy:
            raise ValueError(f"Invalid log level: {level}")
        
        self.default_level = level.upper()
        
        # Apply to root logger
        logging.getLogger().setLevel(getattr(logging, level.upper()))
        
        self.logger.info(
            f"Set default log level to {level}",
            operation="set_default_level",
            level=level
        )
    
    def get_all_levels(self) -> Dict[str, str]:
        """Get all component log levels."""
        return self.component_levels.copy()
    
    def reset_level(self, component: str):
        """Reset component to default level."""
        if component in self.component_levels:
            del self.component_levels[component]
            
            # Apply default level
            logger = logging.getLogger(component)
            logger.setLevel(getattr(logging, self.default_level))
            
            self.logger.info(
                f"Reset log level for {component} to default ({self.default_level})",
                operation="reset_log_level",
                component=component
            )
    
    def should_log(self, component: str, level: str) -> bool:
        """Check if a log should be recorded based on levels."""
        component_level = self.get_level(component)
        
        component_value = self.level_hierarchy.get(component_level, 1)
        log_value = self.level_hierarchy.get(level.upper(), 0)
        
        return log_value >= component_value


class LogFilterManager:
    """Manages log filters and filtering rules."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'log_filter_manager')
        self.filters: Dict[str, LogFilter] = {}
        self.stats = {
            'filters_applied': 0,
            'logs_filtered': 0,
            'logs_modified': 0,
            'logs_routed': 0
        }
    
    def add_filter(self, log_filter: LogFilter):
        """Add a log filter."""
        self.filters[log_filter.name] = log_filter
        
        self.logger.info(
            f"Added log filter: {log_filter.name}",
            operation="add_filter",
            filter_name=log_filter.name,
            action=log_filter.action.value
        )
    
    def remove_filter(self, name: str):
        """Remove a log filter."""
        if name in self.filters:
            del self.filters[name]
            
            self.logger.info(
                f"Removed log filter: {name}",
                operation="remove_filter",
                filter_name=name
            )
    
    def enable_filter(self, name: str):
        """Enable a log filter."""
        if name in self.filters:
            self.filters[name].enabled = True
            
            self.logger.info(
                f"Enabled log filter: {name}",
                operation="enable_filter",
                filter_name=name
            )
    
    def disable_filter(self, name: str):
        """Disable a log filter."""
        if name in self.filters:
            self.filters[name].enabled = False
            
            self.logger.info(
                f"Disabled log filter: {name}",
                operation="disable_filter",
                filter_name=name
            )
    
    def apply_filters(self, log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply all filters to log data."""
        # Sort filters by priority (higher priority first)
        sorted_filters = sorted(
            self.filters.values(),
            key=lambda f: f.priority,
            reverse=True
        )
        
        current_data = log_data
        
        for log_filter in sorted_filters:
            if not log_filter.enabled:
                continue
            
            result = log_filter.apply(current_data)
            
            if result is None:
                # Log was filtered out
                self.stats['logs_filtered'] += 1
                return None
            elif result != current_data:
                # Log was modified
                self.stats['logs_modified'] += 1
                current_data = result
            
            if '_routing' in result:
                self.stats['logs_routed'] += 1
            
            self.stats['filters_applied'] += 1
        
        return current_data
    
    def get_filters(self) -> Dict[str, LogFilter]:
        """Get all filters."""
        return self.filters.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return self.stats.copy()


class LogAnalyzer:
    """Analyzes log patterns and provides insights."""
    
    def __init__(self, window_size: int = 1000):
        self.logger = get_logger(__name__, 'log_analyzer')
        self.window_size = window_size
        self.log_buffer: List[Dict[str, Any]] = []
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_analysis = datetime.utcnow()
    
    def add_log(self, log_data: Dict[str, Any]):
        """Add log to analysis buffer."""
        self.log_buffer.append({
            **log_data,
            '_analyzed_at': datetime.utcnow().isoformat()
        })
        
        # Keep buffer size manageable
        if len(self.log_buffer) > self.window_size:
            self.log_buffer = self.log_buffer[-self.window_size:]
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze log patterns."""
        if not self.log_buffer:
            return {}
        
        # Check cache
        cache_key = 'patterns'
        if self._is_cache_valid(cache_key):
            return self.analysis_cache[cache_key]['data']
        
        analysis = {
            'total_logs': len(self.log_buffer),
            'time_range': self._get_time_range(),
            'level_distribution': self._analyze_levels(),
            'component_activity': self._analyze_components(),
            'error_patterns': self._analyze_errors(),
            'correlation_patterns': self._analyze_correlations(),
            'performance_metrics': self._analyze_performance(),
            'anomalies': self._detect_anomalies()
        }
        
        # Cache results
        self.analysis_cache[cache_key] = {
            'data': analysis,
            'timestamp': datetime.utcnow()
        }
        
        return analysis
    
    def _get_time_range(self) -> Dict[str, str]:
        """Get time range of logs in buffer."""
        if not self.log_buffer:
            return {}
        
        timestamps = [log.get('timestamp', '') for log in self.log_buffer if log.get('timestamp')]
        if not timestamps:
            return {}
        
        return {
            'start': min(timestamps),
            'end': max(timestamps)
        }
    
    def _analyze_levels(self) -> Dict[str, int]:
        """Analyze log level distribution."""
        levels = Counter(log.get('level', 'UNKNOWN') for log in self.log_buffer)
        return dict(levels)
    
    def _analyze_components(self) -> Dict[str, Dict[str, Any]]:
        """Analyze component activity."""
        components = defaultdict(lambda: {'count': 0, 'levels': Counter(), 'operations': Counter()})
        
        for log in self.log_buffer:
            component = log.get('component', 'unknown')
            level = log.get('level', 'UNKNOWN')
            operation = log.get('operation', 'unknown')
            
            components[component]['count'] += 1
            components[component]['levels'][level] += 1
            components[component]['operations'][operation] += 1
        
        # Convert to regular dict
        return {
            comp: {
                'count': data['count'],
                'levels': dict(data['levels']),
                'operations': dict(data['operations'])
            }
            for comp, data in components.items()
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_logs = [log for log in self.log_buffer if log.get('level') in ['ERROR', 'CRITICAL']]
        
        if not error_logs:
            return {'count': 0}
        
        error_messages = Counter(log.get('message', '') for log in error_logs)
        error_components = Counter(log.get('component', 'unknown') for log in error_logs)
        
        return {
            'count': len(error_logs),
            'rate': len(error_logs) / len(self.log_buffer),
            'top_messages': dict(error_messages.most_common(5)),
            'by_component': dict(error_components),
            'recent_errors': error_logs[-5:]  # Last 5 errors
        }
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlation ID patterns."""
        correlations = defaultdict(list)
        
        for log in self.log_buffer:
            correlation_id = log.get('correlation_id', 'unknown')
            if correlation_id != 'unknown':
                correlations[correlation_id].append(log)
        
        # Analyze correlation chains
        correlation_stats = {}
        for corr_id, logs in correlations.items():
            correlation_stats[corr_id] = {
                'log_count': len(logs),
                'duration': self._calculate_duration(logs),
                'components': list(set(log.get('component', 'unknown') for log in logs)),
                'has_errors': any(log.get('level') in ['ERROR', 'CRITICAL'] for log in logs)
            }
        
        return {
            'total_correlations': len(correlations),
            'avg_logs_per_correlation': sum(len(logs) for logs in correlations.values()) / len(correlations) if correlations else 0,
            'correlations_with_errors': sum(1 for stats in correlation_stats.values() if stats['has_errors']),
            'top_correlations': dict(sorted(
                correlation_stats.items(),
                key=lambda x: x[1]['log_count'],
                reverse=True
            )[:5])
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics from logs."""
        # Look for performance-related log entries
        perf_logs = [
            log for log in self.log_buffer
            if any(keyword in log.get('message', '').lower() 
                  for keyword in ['duration', 'time', 'ms', 'seconds', 'performance'])
        ]
        
        return {
            'performance_logs_count': len(perf_logs),
            'performance_logs_rate': len(perf_logs) / len(self.log_buffer) if self.log_buffer else 0,
            'recent_performance_logs': perf_logs[-5:]
        }
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in log patterns."""
        anomalies = []
        
        # Check for error spikes
        recent_logs = self.log_buffer[-100:]  # Last 100 logs
        error_rate = sum(1 for log in recent_logs if log.get('level') in ['ERROR', 'CRITICAL']) / len(recent_logs) if recent_logs else 0
        
        if error_rate > 0.1:  # More than 10% errors
            anomalies.append({
                'type': 'high_error_rate',
                'severity': 'high' if error_rate > 0.2 else 'medium',
                'description': f'High error rate detected: {error_rate:.1%}',
                'value': error_rate
            })
        
        # Check for repeated errors
        error_messages = [log.get('message', '') for log in recent_logs if log.get('level') in ['ERROR', 'CRITICAL']]
        message_counts = Counter(error_messages)
        
        for message, count in message_counts.items():
            if count > 5:  # Same error message more than 5 times
                anomalies.append({
                    'type': 'repeated_error',
                    'severity': 'medium',
                    'description': f'Repeated error detected: "{message[:50]}..." ({count} times)',
                    'value': count,
                    'message': message
                })
        
        return anomalies
    
    def _calculate_duration(self, logs: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate duration between first and last log in a correlation."""
        if len(logs) < 2:
            return None
        
        timestamps = [log.get('timestamp', '') for log in logs if log.get('timestamp')]
        if len(timestamps) < 2:
            return None
        
        try:
            start_time = datetime.fromisoformat(min(timestamps).replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(max(timestamps).replace('Z', '+00:00'))
            return (end_time - start_time).total_seconds()
        except Exception:
            return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.analysis_cache:
            return False
        
        cache_time = self.analysis_cache[cache_key]['timestamp']
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl
    
    def get_insights(self) -> Dict[str, Any]:
        """Get actionable insights from log analysis."""
        patterns = self.analyze_patterns()
        
        insights = {
            'recommendations': [],
            'alerts': [],
            'summary': {}
        }
        
        # Generate recommendations
        error_rate = patterns.get('error_patterns', {}).get('rate', 0)
        if error_rate > 0.05:
            insights['recommendations'].append({
                'type': 'error_rate',
                'priority': 'high' if error_rate > 0.1 else 'medium',
                'message': f'Error rate is {error_rate:.1%}. Consider investigating error patterns.',
                'action': 'Review error logs and implement fixes'
            })
        
        # Generate alerts for anomalies
        anomalies = patterns.get('anomalies', [])
        for anomaly in anomalies:
            if anomaly['severity'] in ['high', 'critical']:
                insights['alerts'].append({
                    'type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'message': anomaly['description'],
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # Summary
        insights['summary'] = {
            'total_logs_analyzed': patterns.get('total_logs', 0),
            'error_rate': error_rate,
            'active_components': len(patterns.get('component_activity', {})),
            'anomalies_detected': len(anomalies),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        return insights


class LogManager:
    """Central log management system."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'log_manager')
        self.level_manager = LogLevelManager()
        self.filter_manager = LogFilterManager()
        self.analyzer = LogAnalyzer()
        
        # Setup default filters
        self._setup_default_filters()
    
    def _setup_default_filters(self):
        """Setup default log filters."""
        # Filter out debug logs in production
        debug_filter = LogFilter(
            name="production_debug_filter",
            description="Filter out debug logs in production",
            conditions=[
                FilterCondition("level", FilterOperator.EQUALS, "DEBUG")
            ],
            action=FilterAction.DENY,
            action_params={},
            priority=100
        )
        
        # Add component tags
        component_tagger = LogFilter(
            name="component_tagger",
            description="Add component tags based on logger name",
            conditions=[
                FilterCondition("logger", FilterOperator.CONTAINS, "axon_interface")
            ],
            action=FilterAction.MODIFY,
            action_params={
                "modifications": {
                    "_add_service": "api",
                    "_add_layer": "interface"
                }
            },
            priority=50
        )
        
        self.filter_manager.add_filter(debug_filter)
        self.filter_manager.add_filter(component_tagger)
    
    def process_log(self, log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a log entry through the management pipeline."""
        # Apply filters
        filtered_data = self.filter_manager.apply_filters(log_data)
        
        if filtered_data is None:
            return None
        
        # Add to analyzer
        self.analyzer.add_log(filtered_data)
        
        return filtered_data
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        return {
            'log_levels': self.level_manager.get_all_levels(),
            'filter_stats': self.filter_manager.get_stats(),
            'analysis': self.analyzer.analyze_patterns(),
            'insights': self.analyzer.get_insights(),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global log manager instance
_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get the global log manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager