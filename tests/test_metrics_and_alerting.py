"""
Comprehensive tests for metrics collection and alerting system.

Tests metrics collection, performance monitoring, business KPIs,
infrastructure monitoring, and alerting functionality.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import metrics and alerting components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.metrics_collector import (
    MetricsCollector, Counter, Gauge, Histogram, Timer, MetricType, MetricUnit,
    MetricValue, MetricSeries, get_metrics_collector, increment_counter,
    set_gauge, observe_histogram, time_operation
)
from shared.alerting_system import (
    AlertManager, AlertRule, Alert, AlertSeverity, AlertStatus, NotificationChannel,
    NotificationChannelConfig, NotificationSender, EscalationPolicy,
    get_alert_manager, create_alert, setup_email_notifications
)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_metric_value_creation(self):
        """Test MetricValue creation and serialization."""
        timestamp = datetime.utcnow()
        metric_value = MetricValue(
            name='test_metric',
            value=42.5,
            metric_type=MetricType.GAUGE,
            unit=MetricUnit.COUNT,
            timestamp=timestamp,
            tags={'service': 'test'},
            labels={'endpoint': '/api/test'}
        )
        
        assert metric_value.name == 'test_metric'
        assert metric_value.value == 42.5
        assert metric_value.metric_type == MetricType.GAUGE
        assert metric_value.unit == MetricUnit.COUNT
        assert metric_value.tags['service'] == 'test'
        assert metric_value.labels['endpoint'] == '/api/test'
        
        # Test dictionary conversion
        metric_dict = metric_value.to_dict()
        assert metric_dict['name'] == 'test_metric'
        assert metric_dict['value'] == 42.5
        assert metric_dict['type'] == 'gauge'
        assert metric_dict['unit'] == 'count'
    
    def test_metric_series(self):
        """Test MetricSeries functionality."""
        series = MetricSeries(
            name='test_series',
            metric_type=MetricType.COUNTER,
            unit=MetricUnit.COUNT,
            tags={'service': 'test'}
        )
        
        # Add values
        series.add_value(10, labels={'endpoint': '/api/v1'})
        series.add_value(15, labels={'endpoint': '/api/v2'})
        series.add_value(20, labels={'endpoint': '/api/v1'})
        
        assert len(series.values) == 3
        
        # Test latest value
        latest = series.get_latest_value()
        assert latest.value == 20
        assert latest.labels['endpoint'] == '/api/v1'
        
        # Test statistics
        stats = series.calculate_statistics(window_minutes=60)
        assert stats['count'] == 3
        assert stats['sum'] == 45
        assert stats['min'] == 10
        assert stats['max'] == 20
        assert stats['mean'] == 15
    
    def test_counter(self):
        """Test Counter metric."""
        counter = Counter('test_counter', 'Test counter metric')
        
        # Test initial value
        assert counter.get_value() == 0
        
        # Test increment
        counter.increment()
        assert counter.get_value() == 1
        
        counter.increment(5)
        assert counter.get_value() == 6
        
        # Test with labels
        counter.increment(2, endpoint='/api/test')
        assert counter.get_value() == 8
        
        # Test reset
        counter.reset()
        assert counter.get_value() == 0
    
    def test_gauge(self):
        """Test Gauge metric."""
        gauge = Gauge('test_gauge', 'Test gauge metric', MetricUnit.PERCENT)
        
        # Test initial value
        assert gauge.get_value() == 0
        
        # Test set
        gauge.set(75.5)
        assert gauge.get_value() == 75.5
        
        # Test increment
        gauge.increment(10)
        assert gauge.get_value() == 85.5
        
        # Test decrement
        gauge.decrement(5.5)
        assert gauge.get_value() == 80
        
        # Test with labels
        gauge.set(90, service='test')
        assert gauge.get_value() == 90
    
    def test_histogram(self):
        """Test Histogram metric."""
        histogram = Histogram(
            'test_histogram',
            'Test histogram metric',
            MetricUnit.SECONDS,
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, float('inf')]
        )
        
        # Observe values
        values = [0.05, 0.3, 0.8, 2.5, 7.2, 15.0]
        for value in values:
            histogram.observe(value)
        
        # Test statistics
        stats = histogram.get_statistics()
        assert stats['count'] == 6
        assert stats['sum'] == sum(values)
        assert stats['mean'] == sum(values) / len(values)
        
        # Test bucket counts
        buckets = stats['buckets']
        assert buckets[0.1] == 1  # 0.05
        assert buckets[0.5] == 2  # 0.05, 0.3
        assert buckets[1.0] == 3  # 0.05, 0.3, 0.8
        assert buckets[5.0] == 4  # 0.05, 0.3, 0.8, 2.5
        assert buckets[10.0] == 5  # 0.05, 0.3, 0.8, 2.5, 7.2
        assert buckets[float('inf')] == 6  # all values
    
    def test_timer(self):
        """Test Timer metric."""
        timer = Timer('test_timer', 'Test timer metric')
        
        # Test context manager
        with timer.time(operation='test_op'):
            time.sleep(0.1)  # Sleep for 100ms
        
        # Check that duration was recorded
        stats = timer.histogram.get_statistics()
        assert stats['count'] == 1
        assert 0.09 <= stats['sum'] <= 0.15  # Allow some variance
        
        # Test direct recording
        timer.record(0.5, operation='manual')
        stats = timer.histogram.get_statistics()
        assert stats['count'] == 2
    
    def test_metrics_collector(self):
        """Test MetricsCollector functionality."""
        collector = MetricsCollector()
        
        # Test recording metrics
        collector.record_metric(
            'test_metric',
            42,
            MetricType.GAUGE,
            MetricUnit.COUNT,
            tags={'service': 'test'}
        )
        
        # Test getting metric series
        series = collector.get_metric_series('test_metric', {'service': 'test'})
        assert series is not None
        assert series.name == 'test_metric'
        assert len(series.values) == 1
        
        # Test getting counter
        counter = collector.get_counter('test_counter', tags={'service': 'test'})
        counter.increment(5)
        assert counter.get_value() == 5
        
        # Test getting gauge
        gauge = collector.get_gauge('test_gauge', unit=MetricUnit.PERCENT)
        gauge.set(75)
        assert gauge.get_value() == 75
        
        # Test metrics summary
        summary = collector.get_metrics_summary()
        assert 'total_metrics' in summary
        assert 'collection_stats' in summary
        assert 'metrics' in summary
        assert summary['total_metrics'] > 0
    
    def test_business_metrics(self):
        """Test business metrics recording."""
        collector = MetricsCollector()
        
        # Test job metrics
        collector.record_job_created('scrape', 'high')
        collector.record_job_completed('scrape', 2.5, success=True)
        collector.record_job_completed('scrape', 1.0, success=False)
        
        # Test article metrics
        collector.record_article_scraped('news_site', success=True)
        collector.record_article_scraped('blog', success=False)
        
        # Test API metrics
        collector.record_api_call('/api/scrape', 'premium')
        collector.record_api_call('/api/search', 'free')
        
        # Test gauge metrics
        collector.set_active_users(150)
        collector.set_queue_size('scraping', 25)
        collector.set_cache_hit_rate('redis', 0.85)
        
        # Verify metrics were recorded
        summary = collector.get_metrics_summary()
        assert summary['total_metrics'] > 0
    
    def test_prometheus_export(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_metric('test_counter', 10, MetricType.COUNTER, MetricUnit.COUNT)
        collector.record_metric('test_gauge', 75.5, MetricType.GAUGE, MetricUnit.PERCENT)
        
        # Export in Prometheus format
        prometheus_output = collector.export_metrics('prometheus')
        
        assert '# HELP test_counter' in prometheus_output
        assert '# TYPE test_counter counter' in prometheus_output
        assert 'test_counter 10' in prometheus_output
        
        assert '# HELP test_gauge' in prometheus_output
        assert '# TYPE test_gauge gauge' in prometheus_output
        assert 'test_gauge 75.5' in prometheus_output
    
    def test_json_export(self):
        """Test JSON format export."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_metric('test_metric', 42, MetricType.GAUGE, MetricUnit.COUNT)
        
        # Export in JSON format
        json_output = collector.export_metrics('json')
        parsed = json.loads(json_output)
        
        assert 'total_metrics' in parsed
        assert 'metrics' in parsed
        assert parsed['total_metrics'] > 0
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test system metrics collection."""
        collector = MetricsCollector()
        
        # Start system metrics collection
        await collector.start_system_metrics_collection()
        
        # Wait for at least one collection cycle
        await asyncio.sleep(2)
        
        # Stop collection
        await collector.stop_system_metrics_collection()
        
        # Check that system metrics were collected
        summary = collector.get_metrics_summary()
        metric_names = [info['name'] for info in summary['metrics'].values()]
        
        # Should have collected CPU, memory, disk metrics
        assert any('cpu' in name for name in metric_names)
        assert any('memory' in name for name in metric_names)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test increment_counter
        increment_counter('test_counter_func', 5, service='test')
        
        # Test set_gauge
        set_gauge('test_gauge_func', 42.5, MetricUnit.PERCENT, service='test')
        
        # Test observe_histogram
        observe_histogram('test_histogram_func', 1.5, MetricUnit.SECONDS, endpoint='/api')
        
        # Test time_operation
        with time_operation('test_timer_func', operation='test'):
            time.sleep(0.01)
        
        # Verify metrics were recorded
        collector = get_metrics_collector()
        summary = collector.get_metrics_summary()
        assert summary['total_metrics'] > 0


class TestAlertingSystem:
    """Test alerting system functionality."""
    
    def test_alert_rule_creation(self):
        """Test AlertRule creation and evaluation."""
        rule = AlertRule(
            id='test_rule',
            name='Test Rule',
            description='Test alert rule',
            metric_name='test_metric',
            condition='> 80',
            threshold=80,
            severity=AlertSeverity.HIGH,
            duration_minutes=5
        )
        
        assert rule.id == 'test_rule'
        assert rule.name == 'Test Rule'
        assert rule.severity == AlertSeverity.HIGH
        
        # Test evaluation
        assert rule.evaluate(85) is True
        assert rule.evaluate(75) is False
        assert rule.evaluate(80) is False  # Not greater than
        
        # Test different conditions
        rule.condition = '>= 80'
        assert rule.evaluate(80) is True
        
        rule.condition = '< 50'
        assert rule.evaluate(45) is True
        assert rule.evaluate(55) is False
        
        rule.condition = '== 100'
        assert rule.evaluate(100) is True
        assert rule.evaluate(99) is False
    
    def test_alert_creation(self):
        """Test Alert creation and management."""
        alert = Alert(
            id='test_alert',
            rule_id='test_rule',
            rule_name='Test Rule',
            metric_name='test_metric',
            current_value=85,
            threshold=80,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message='Test alert message',
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert alert.is_active() is True
        assert alert.get_duration().total_seconds() >= 0
        
        # Test acknowledgment
        alert.acknowledge('test_user')
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == 'test_user'
        assert alert.acknowledged_at is not None
        
        # Test resolution
        alert.resolve()
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
        assert alert.is_active() is False
    
    def test_escalation_policy(self):
        """Test EscalationPolicy functionality."""
        policy = EscalationPolicy(
            id='test_policy',
            name='Test Policy',
            description='Test escalation policy',
            levels=[
                {'delay_minutes': 0, 'channels': ['email']},
                {'delay_minutes': 15, 'channels': ['slack']},
                {'delay_minutes': 60, 'channels': ['pagerduty']}
            ]
        )
        
        # Test escalation level calculation
        assert policy.get_escalation_level(5) == 0   # First level
        assert policy.get_escalation_level(20) == 1  # Second level
        assert policy.get_escalation_level(90) == 2  # Third level
    
    @pytest.mark.asyncio
    async def test_notification_sender(self):
        """Test NotificationSender functionality."""
        sender = NotificationSender()
        await sender.start()
        
        # Create test alert
        alert = Alert(
            id='test_alert',
            rule_id='test_rule',
            rule_name='Test Alert',
            metric_name='test_metric',
            current_value=85,
            threshold=80,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message='Test alert message',
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Test webhook notification (mock)
        webhook_config = NotificationChannelConfig(
            channel=NotificationChannel.WEBHOOK,
            config={'url': 'https://httpbin.org/post'}
        )
        
        with patch.object(sender.session, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            success = await sender.send_notification(alert, webhook_config)
            assert success is True
            assert len(alert.notifications_sent) == 1
        
        await sender.stop()
    
    @pytest.mark.asyncio
    async def test_alert_manager(self):
        """Test AlertManager functionality."""
        # Create a separate metrics collector for testing
        metrics_collector = MetricsCollector()
        
        alert_manager = AlertManager()
        alert_manager.metrics_collector = metrics_collector
        
        # Add a test rule
        rule = AlertRule(
            id='test_cpu_rule',
            name='Test CPU Rule',
            description='Test CPU usage alert',
            metric_name='cpu_usage',
            condition='> 80',
            threshold=80,
            severity=AlertSeverity.HIGH,
            duration_minutes=1,  # Short duration for testing
            notification_channels=[NotificationChannel.EMAIL]
        )
        
        alert_manager.add_alert_rule(rule)
        
        # Add notification channel
        email_config = NotificationChannelConfig(
            channel=NotificationChannel.EMAIL,
            config={
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'from_email': 'test@example.com',
                'to_emails': ['admin@example.com']
            }
        )
        alert_manager.add_notification_channel('email', email_config)
        
        # Record metric that should trigger alert
        metrics_collector.record_metric(
            'cpu_usage', 85, MetricType.GAUGE, MetricUnit.PERCENT
        )
        
        # Manually evaluate rules (normally done by background task)
        await alert_manager._evaluate_all_rules()
        
        # Wait for duration threshold
        await asyncio.sleep(1.1)
        
        # Evaluate again
        await alert_manager._evaluate_all_rules()
        
        # Check if alert was created
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Test alert acknowledgment
        alert_id = active_alerts[0].id
        success = alert_manager.acknowledge_alert(alert_id, 'test_user')
        assert success is True
        
        # Record metric that resolves the alert
        metrics_collector.record_metric(
            'cpu_usage', 70, MetricType.GAUGE, MetricUnit.PERCENT
        )
        
        # Evaluate rules again
        await alert_manager._evaluate_all_rules()
        
        # Check that alert was resolved
        active_alerts = alert_manager.get_active_alerts()
        # Alert should still be active but acknowledged
        assert len(active_alerts) == 0 or active_alerts[0].status == AlertStatus.ACKNOWLEDGED
    
    def test_alert_manager_stats(self):
        """Test AlertManager statistics."""
        alert_manager = AlertManager()
        
        stats = alert_manager.get_stats()
        
        assert 'alerts_created' in stats
        assert 'alerts_resolved' in stats
        assert 'notifications_sent' in stats
        assert 'rules_evaluated' in stats
        assert 'active_alerts_total' in stats
        assert 'total_rules' in stats
        assert 'enabled_rules' in stats
    
    def test_default_alert_rules(self):
        """Test that default alert rules are created."""
        alert_manager = AlertManager()
        
        # Should have default rules
        assert len(alert_manager.alert_rules) > 0
        
        # Check for specific default rules
        rule_names = [rule.name for rule in alert_manager.alert_rules.values()]
        assert 'High CPU Usage' in rule_names
        assert 'High Memory Usage' in rule_names
        assert 'Low Disk Space' in rule_names
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions."""
        # Test create_alert
        rule_id = await create_alert(
            rule_name='Test Custom Alert',
            metric_name='custom_metric',
            condition='> 100',
            threshold=100,
            severity=AlertSeverity.MEDIUM,
            description='Custom test alert',
            duration_minutes=2
        )
        
        assert rule_id.startswith('custom_')
        
        # Verify rule was added
        alert_manager = get_alert_manager()
        assert rule_id in alert_manager.alert_rules
        
        rule = alert_manager.alert_rules[rule_id]
        assert rule.name == 'Test Custom Alert'
        assert rule.severity == AlertSeverity.MEDIUM


@pytest.mark.asyncio
async def test_end_to_end_monitoring():
    """Test end-to-end monitoring and alerting."""
    # Setup metrics collector
    metrics_collector = get_metrics_collector()
    
    # Setup alert manager
    alert_manager = get_alert_manager()
    
    # Create a test alert rule
    rule = AlertRule(
        id='e2e_test_rule',
        name='E2E Test Rule',
        description='End-to-end test alert',
        metric_name='e2e_test_metric',
        condition='> 50',
        threshold=50,
        severity=AlertSeverity.MEDIUM,
        duration_minutes=1
    )
    
    alert_manager.add_alert_rule(rule)
    
    # Record metrics over time
    for i in range(5):
        value = 40 + (i * 5)  # 40, 45, 50, 55, 60
        metrics_collector.record_metric(
            'e2e_test_metric', value, MetricType.GAUGE, MetricUnit.COUNT
        )
        
        # Evaluate rules
        await alert_manager._evaluate_all_rules()
        
        if i >= 3:  # Values 55 and 60 should trigger alert
            await asyncio.sleep(0.2)
    
    # Wait for duration threshold
    await asyncio.sleep(1.1)
    
    # Final evaluation
    await alert_manager._evaluate_all_rules()
    
    # Check results
    active_alerts = alert_manager.get_active_alerts()
    metrics_summary = metrics_collector.get_metrics_summary()
    alert_stats = alert_manager.get_stats()
    
    # Verify metrics were collected
    assert metrics_summary['total_metrics'] > 0
    
    # Verify alert system is working
    assert alert_stats['rules_evaluated'] > 0
    
    # Clean up
    alert_manager.remove_alert_rule('e2e_test_rule')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])