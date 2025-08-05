#!/usr/bin/env python3
"""
Comprehensive test suite for Project Synapse monitoring, logging, and observability system.

Tests all components of Task 16: logging, metrics collection, and alerting.
"""

import asyncio
import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

# Import all monitoring components
from shared.logging_config import (
    LoggingConfig, SynapseLogger, get_logger, CorrelationContext,
    JSONFormatter, ColoredFormatter, CorrelationFilter
)
from shared.log_aggregator import (
    LogAggregator, LogDestinationConfig, LogDestination, LogLevel,
    LogEntry, LogBatch, LogFilter, setup_log_aggregation
)
from shared.log_manager import (
    LogManager, LogLevelManager, LogFilterManager, LogAnalyzer,
    LogFilter as ManagerLogFilter, FilterCondition, FilterOperator, FilterAction
)
from shared.metrics_collector import (
    MetricsCollector, Counter, Gauge, Histogram, Timer,
    MetricType, MetricUnit, get_metrics_collector,
    increment_counter, set_gauge, observe_histogram, time_operation
)
from shared.alerting_system import (
    AlertManager, AlertRule, Alert, AlertSeverity, AlertStatus,
    NotificationChannel, NotificationChannelConfig, NotificationSender,
    EscalationPolicy, get_alert_manager, create_alert
)


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test_pass(self, test_name: str):
        """Record a passing test."""
        print(f"✅ {test_name}")
        self.passed += 1
    
    def test_fail(self, test_name: str, error: str):
        """Record a failing test."""
        print(f"❌ {test_name}: {error}")
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/total*100):.1f}%" if total > 0 else "No tests run")
        
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        
        return self.failed == 0


def test_logging_system(results: TestResults):
    """Test comprehensive logging system."""
    print(f"\n{'='*60}")
    print("TESTING LOGGING SYSTEM")
    print(f"{'='*60}")
    
    try:
        # Test SynapseLogger creation
        logger = get_logger('test.logging', 'logging_component')
        if isinstance(logger, SynapseLogger):
            results.test_pass("SynapseLogger creation")
        else:
            results.test_fail("SynapseLogger creation", "Invalid logger type")
        
        # Test correlation context
        with CorrelationContext('test-correlation-123', 'test-user-456'):
            logger.info('Test message with correlation', operation='test_correlation')
            results.test_pass("Correlation context logging")
        
        # Test different log levels
        logger.debug('Debug message', operation='test_levels')
        logger.info('Info message', operation='test_levels')
        logger.warning('Warning message', operation='test_levels')
        logger.error('Error message', operation='test_levels')
        results.test_pass("Multiple log levels")
        
        # Test JSON formatter
        formatter = JSONFormatter()
        import logging
        record = logging.LogRecord(
            name='test.logger', level=logging.INFO, pathname='test.py',
            lineno=42, msg='Test message', args=(), exc_info=None
        )
        record.correlation_id = 'test-correlation'
        record.user_id = 'test-user'
        record.component = 'test_component'
        record.operation = 'test_operation'
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        if parsed['level'] == 'INFO' and parsed['correlation_id'] == 'test-correlation':
            results.test_pass("JSON formatter")
        else:
            results.test_fail("JSON formatter", "Invalid format")
        
        # Test log level manager
        level_manager = LogLevelManager()
        level_manager.set_level('test.component', 'DEBUG')
        if level_manager.get_level('test.component') == 'DEBUG':
            results.test_pass("Log level management")
        else:
            results.test_fail("Log level management", "Level not set correctly")
        
        # Test log filter manager
        filter_manager = LogFilterManager()
        
        # Create a filter condition
        condition = FilterCondition('level', FilterOperator.EQUALS, 'ERROR')
        log_filter = ManagerLogFilter(
            name='error_filter',
            description='Filter error logs',
            conditions=[condition],
            action=FilterAction.ALLOW,
            action_params={}
        )
        
        filter_manager.add_filter(log_filter)
        
        # Test filter application
        log_data = {'level': 'ERROR', 'message': 'Test error'}
        result = filter_manager.apply_filters(log_data)
        if result == log_data:
            results.test_pass("Log filtering")
        else:
            results.test_fail("Log filtering", "Filter not applied correctly")
        
        # Test log analyzer
        analyzer = LogAnalyzer(window_size=100)
        for i in range(10):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'INFO' if i % 3 != 0 else 'ERROR',
                'logger': 'test.logger',
                'message': f'Test message {i}',
                'correlation_id': f'corr-{i // 3}',
                'component': 'test_component',
                'operation': f'operation_{i % 3}'
            }
            analyzer.add_log(log_data)
        
        patterns = analyzer.analyze_patterns()
        if patterns['total_logs'] == 10 and 'level_distribution' in patterns:
            results.test_pass("Log analysis")
        else:
            results.test_fail("Log analysis", "Analysis failed")
        
        # Test log manager integration
        log_manager = LogManager()
        test_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO',
            'logger': 'test.logger',
            'message': 'Integration test message',
            'correlation_id': 'integration-test',
            'user_id': 'test-user',
            'component': 'test_component',
            'operation': 'integration_test'
        }
        
        processed = log_manager.process_log(test_log)
        if processed is not None:
            results.test_pass("Log manager integration")
        else:
            results.test_fail("Log manager integration", "Processing failed")
    
    except Exception as e:
        results.test_fail("Logging system test", str(e))


def test_metrics_system(results: TestResults):
    """Test comprehensive metrics system."""
    print(f"\n{'='*60}")
    print("TESTING METRICS SYSTEM")
    print(f"{'='*60}")
    
    try:
        # Test metrics collector singleton
        collector1 = get_metrics_collector()
        collector2 = MetricsCollector.get_instance()
        if collector1 is collector2:
            results.test_pass("Metrics collector singleton")
        else:
            results.test_fail("Metrics collector singleton", "Not singleton")
        
        # Test counter metrics
        counter = collector1.get_counter('test_counter', 'Test counter')
        counter.increment(5)
        counter.increment(3, endpoint='/api/test')
        if counter.get_value() == 8:
            results.test_pass("Counter metrics")
        else:
            results.test_fail("Counter metrics", f"Expected 8, got {counter.get_value()}")
        
        # Test gauge metrics
        gauge = collector1.get_gauge('test_gauge', 'Test gauge', MetricUnit.PERCENT)
        gauge.set(75.5)
        gauge.increment(10)
        gauge.decrement(5.5)
        if gauge.get_value() == 80:
            results.test_pass("Gauge metrics")
        else:
            results.test_fail("Gauge metrics", f"Expected 80, got {gauge.get_value()}")
        
        # Test histogram metrics
        histogram = collector1.get_histogram('test_histogram', 'Test histogram', MetricUnit.SECONDS)
        values = [0.1, 0.5, 1.2, 2.8, 0.3]
        for value in values:
            histogram.observe(value)
        
        stats = histogram.get_statistics()
        if stats['count'] == 5 and abs(stats['sum'] - sum(values)) < 0.001:
            results.test_pass("Histogram metrics")
        else:
            results.test_fail("Histogram metrics", f"Stats incorrect: {stats}")
        
        # Test timer metrics
        timer = collector1.get_timer('test_timer', 'Test timer')
        with timer.time(operation='test_timing'):
            time.sleep(0.01)  # 10ms
        
        timer_stats = timer.histogram.get_statistics()
        if timer_stats['count'] == 1 and timer_stats['sum'] > 0.005:  # At least 5ms
            results.test_pass("Timer metrics")
        else:
            results.test_fail("Timer metrics", f"Timer stats incorrect: {timer_stats}")
        
        # Test business metrics
        collector1.record_job_created('scrape', 'high')
        collector1.record_job_completed('scrape', 2.5, success=True)
        collector1.record_article_scraped('news_site', success=True)
        collector1.record_api_call('/api/scrape', 'premium')
        collector1.set_active_users(150)
        collector1.set_queue_size('scraping', 25)
        collector1.set_cache_hit_rate('redis', 0.85)
        results.test_pass("Business metrics recording")
        
        # Test metrics export
        json_export = collector1.export_metrics('json')
        parsed_export = json.loads(json_export)
        if 'total_metrics' in parsed_export and parsed_export['total_metrics'] > 0:
            results.test_pass("JSON metrics export")
        else:
            results.test_fail("JSON metrics export", "Export failed")
        
        prometheus_export = collector1.export_metrics('prometheus')
        if '# HELP' in prometheus_export and '# TYPE' in prometheus_export:
            results.test_pass("Prometheus metrics export")
        else:
            results.test_fail("Prometheus metrics export", "Export failed")
        
        # Test convenience functions
        increment_counter('convenience_counter', 3, service='test')
        set_gauge('convenience_gauge', 42.5, MetricUnit.PERCENT, service='test')
        observe_histogram('convenience_histogram', 1.5, MetricUnit.SECONDS, endpoint='/api')
        
        with time_operation('convenience_timer', operation='test'):
            time.sleep(0.005)
        
        results.test_pass("Convenience functions")
        
        # Test metrics summary
        summary = collector1.get_metrics_summary()
        if 'total_metrics' in summary and 'collection_stats' in summary:
            results.test_pass("Metrics summary")
        else:
            results.test_fail("Metrics summary", "Summary incomplete")
    
    except Exception as e:
        results.test_fail("Metrics system test", str(e))


async def test_alerting_system(results: TestResults):
    """Test comprehensive alerting system."""
    print(f"\n{'='*60}")
    print("TESTING ALERTING SYSTEM")
    print(f"{'='*60}")
    
    try:
        # Test alert rule creation and evaluation
        rule = AlertRule(
            id='test_alert_rule',
            name='Test Alert Rule',
            description='Test alert for high CPU usage',
            metric_name='cpu_usage',
            condition='> 80',
            threshold=80,
            severity=AlertSeverity.HIGH,
            duration_minutes=1
        )
        
        if rule.evaluate(85) and not rule.evaluate(75):
            results.test_pass("Alert rule evaluation")
        else:
            results.test_fail("Alert rule evaluation", "Evaluation logic incorrect")
        
        # Test alert creation
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
        
        if alert.is_active():
            results.test_pass("Alert creation")
        else:
            results.test_fail("Alert creation", "Alert not active")
        
        # Test alert acknowledgment
        alert.acknowledge('test_user')
        if alert.status == AlertStatus.ACKNOWLEDGED and alert.acknowledged_by == 'test_user':
            results.test_pass("Alert acknowledgment")
        else:
            results.test_fail("Alert acknowledgment", "Acknowledgment failed")
        
        # Test alert resolution
        alert.resolve()
        if alert.status == AlertStatus.RESOLVED and not alert.is_active():
            results.test_pass("Alert resolution")
        else:
            results.test_fail("Alert resolution", "Resolution failed")
        
        # Test escalation policy
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
        
        if (policy.get_escalation_level(5) == 0 and 
            policy.get_escalation_level(20) == 1 and 
            policy.get_escalation_level(90) == 2):
            results.test_pass("Escalation policy")
        else:
            results.test_fail("Escalation policy", "Escalation levels incorrect")
        
        # Test notification sender
        sender = NotificationSender()
        await sender.start()
        
        # Test webhook notification config
        webhook_config = NotificationChannelConfig(
            channel=NotificationChannel.WEBHOOK,
            config={'url': 'https://httpbin.org/post'}
        )
        
        # Mock the session for testing
        class MockResponse:
            status = 200
        
        class MockSession:
            async def post(self, *args, **kwargs):
                class MockContext:
                    async def __aenter__(self):
                        return MockResponse()
                    async def __aexit__(self, *args):
                        pass
                return MockContext()
        
        sender.session = MockSession()
        
        test_alert = Alert(
            id='notification_test_alert',
            rule_id='test_rule',
            rule_name='Notification Test',
            metric_name='test_metric',
            current_value=85,
            threshold=80,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message='Test notification',
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        success = await sender.send_notification(test_alert, webhook_config)
        if success:
            results.test_pass("Notification sending")
        else:
            results.test_fail("Notification sending", "Notification failed")
        
        await sender.stop()
        
        # Test alert manager
        alert_manager = get_alert_manager()
        
        # Test adding alert rule
        test_rule = AlertRule(
            id='manager_test_rule',
            name='Manager Test Rule',
            description='Test rule for manager',
            metric_name='test_manager_metric',
            condition='> 50',
            threshold=50,
            severity=AlertSeverity.MEDIUM,
            duration_minutes=1
        )
        
        alert_manager.add_alert_rule(test_rule)
        if 'manager_test_rule' in alert_manager.alert_rules:
            results.test_pass("Alert manager rule addition")
        else:
            results.test_fail("Alert manager rule addition", "Rule not added")
        
        # Test alert manager stats
        stats = alert_manager.get_stats()
        if 'alerts_created' in stats and 'total_rules' in stats:
            results.test_pass("Alert manager statistics")
        else:
            results.test_fail("Alert manager statistics", "Stats incomplete")
        
        # Test convenience function
        rule_id = await create_alert(
            rule_name='Convenience Test Alert',
            metric_name='convenience_metric',
            condition='> 100',
            threshold=100,
            severity=AlertSeverity.LOW,
            description='Test convenience function'
        )
        
        if rule_id and rule_id in alert_manager.alert_rules:
            results.test_pass("Alert creation convenience function")
        else:
            results.test_fail("Alert creation convenience function", "Function failed")
        
        # Clean up test rules
        alert_manager.remove_alert_rule('manager_test_rule')
        alert_manager.remove_alert_rule(rule_id)
    
    except Exception as e:
        results.test_fail("Alerting system test", str(e))


async def test_integration(results: TestResults):
    """Test integration between all monitoring components."""
    print(f"\n{'='*60}")
    print("TESTING SYSTEM INTEGRATION")
    print(f"{'='*60}")
    
    try:
        # Test logging with metrics
        logger = get_logger('integration.test', 'integration_component')
        collector = get_metrics_collector()
        
        # Record some metrics
        with CorrelationContext('integration-test-123', 'integration-user'):
            logger.info('Starting integration test', operation='integration_test')
            
            # Record metrics
            collector.record_job_created('integration_test', 'normal')
            collector.record_api_call('/api/integration', 'premium')
            
            # Use convenience functions
            increment_counter('integration_counter', 1, test='integration')
            set_gauge('integration_gauge', 100, test='integration')
            
            with time_operation('integration_timer', test='integration'):
                time.sleep(0.01)
            
            logger.info('Integration test metrics recorded', operation='integration_test')
        
        results.test_pass("Logging and metrics integration")
        
        # Test metrics with alerting
        alert_manager = get_alert_manager()
        
        # Create an alert rule for integration metric
        integration_rule = AlertRule(
            id='integration_alert_rule',
            name='Integration Alert Rule',
            description='Alert for integration testing',
            metric_name='integration_test_metric',
            condition='> 75',
            threshold=75,
            severity=AlertSeverity.MEDIUM,
            duration_minutes=1
        )
        
        alert_manager.add_alert_rule(integration_rule)
        
        # Record metric that should trigger alert
        collector.record_metric(
            'integration_test_metric', 80, MetricType.GAUGE, MetricUnit.PERCENT
        )
        
        # Manually evaluate rules (normally done by background task)
        await alert_manager._evaluate_all_rules()
        
        results.test_pass("Metrics and alerting integration")
        
        # Test comprehensive stats
        log_manager = LogManager()
        comprehensive_stats = log_manager.get_comprehensive_stats()
        metrics_summary = collector.get_metrics_summary()
        alert_stats = alert_manager.get_stats()
        
        if (comprehensive_stats and metrics_summary and alert_stats and
            'analysis' in comprehensive_stats and 
            'total_metrics' in metrics_summary and
            'total_rules' in alert_stats):
            results.test_pass("Comprehensive monitoring statistics")
        else:
            results.test_fail("Comprehensive monitoring statistics", "Stats incomplete")
        
        # Clean up
        alert_manager.remove_alert_rule('integration_alert_rule')
        
        # Test end-to-end monitoring workflow
        with CorrelationContext('e2e-workflow-test', 'e2e-user'):
            logger.info('Starting E2E workflow test', operation='e2e_workflow')
            
            # Simulate application workflow
            collector.record_request('POST', '/api/scrape', 200, 0.5)
            collector.record_job_created('scrape', 'high')
            
            # Simulate processing
            time.sleep(0.01)
            
            collector.record_job_completed('scrape', 0.8, success=True)
            collector.record_article_scraped('test_source', success=True)
            
            logger.info('E2E workflow completed successfully', operation='e2e_workflow')
        
        results.test_pass("End-to-end monitoring workflow")
    
    except Exception as e:
        results.test_fail("Integration test", str(e))


async def main():
    """Run all monitoring system tests."""
    print("🧠 PROJECT SYNAPSE - MONITORING SYSTEM TEST SUITE")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    
    results = TestResults()
    
    # Run test suites
    test_logging_system(results)
    test_metrics_system(results)
    await test_alerting_system(results)
    await test_integration(results)
    
    # Print summary
    success = results.summary()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED! Monitoring, logging, and observability system is working correctly.")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED! Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)