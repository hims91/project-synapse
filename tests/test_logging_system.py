"""
Comprehensive tests for the logging system.

Tests structured logging, correlation IDs, log aggregation, filtering,
and log analysis functionality.
"""

import pytest
import asyncio
import json
import logging
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import logging system components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.logging_config import (
    LoggingConfig, SynapseLogger, CorrelationContext, JSONFormatter,
    ColoredFormatter, CorrelationFilter, get_logger, set_correlation_id,
    get_correlation_id
)
from shared.log_aggregator import (
    LogAggregator, LogForwarder, LogDestinationConfig, LogDestination,
    LogLevel, LogEntry, LogBatch, LogFilter as AggregatorLogFilter
)
from shared.log_manager import (
    LogManager, LogLevelManager, LogFilterManager, LogAnalyzer,
    LogFilter, FilterCondition, FilterOperator, FilterAction
)


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_synapse_logger_creation(self):
        """Test SynapseLogger creation and basic functionality."""
        logger = SynapseLogger('test.logger', 'test_component')
        
        assert logger.component == 'test_component'
        assert logger.logger.name == 'test.logger'
    
    def test_correlation_context(self):
        """Test correlation context management."""
        correlation_id_value = 'test-correlation-123'
        user_id_value = 'test-user-456'
        
        with CorrelationContext(correlation_id_value, user_id_value):
            assert get_correlation_id() == correlation_id_value
        
        # Context should be cleared after exiting
        assert get_correlation_id() is None
    
    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name='test.logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Add correlation context
        record.correlation_id = 'test-correlation'
        record.user_id = 'test-user'
        record.request_id = 'test-request'
        record.component = 'test_component'
        record.operation = 'test_operation'
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed['level'] == 'INFO'
        assert parsed['message'] == 'Test message'
        assert parsed['correlation_id'] == 'test-correlation'
        assert parsed['user_id'] == 'test-user'
        assert parsed['component'] == 'test_component'
        assert 'timestamp' in parsed
    
    def test_colored_formatter(self):
        """Test colored formatter."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        record = logging.LogRecord(
            name='test.logger',
            level=logging.ERROR,
            pathname='test.py',
            lineno=42,
            msg='Test error message',
            args=(),
            exc_info=None
        )
        
        record.correlation_id = 'test-correlation'
        record.user_id = 'test-user'
        
        formatted = formatter.format(record)
        
        # Should contain ANSI color codes for ERROR level
        assert '\033[31m' in formatted  # Red color for ERROR
        assert 'Test error message' in formatted
        assert '[test-cor]' in formatted  # Truncated correlation ID
    
    def test_correlation_filter(self):
        """Test correlation filter."""
        correlation_filter = CorrelationFilter()
        
        record = logging.LogRecord(
            name='test.logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Set correlation context
        with CorrelationContext('test-correlation', 'test-user', 'test-request'):
            result = correlation_filter.filter(record)
            
            assert result is True
            assert record.correlation_id == 'test-correlation'
            assert record.user_id == 'test-user'
            assert record.request_id == 'test-request'
    
    def test_logging_setup(self):
        """Test logging setup with different configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            
            LoggingConfig.setup_logging(
                level=logging.DEBUG,
                format_type='json',
                log_file=log_file,
                console_output=False,
                correlation_tracking=True
            )
            
            # Test that log file is created
            logger = get_logger('test.setup')
            logger.info('Test log message', operation='test_setup')
            
            assert os.path.exists(log_file)
            
            # Read and verify log content
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert 'Test log message' in log_content
                
                # Parse JSON log entry
                log_entry = json.loads(log_content.strip())
                assert log_entry['level'] == 'INFO'
                assert log_entry['operation'] == 'test_setup'


class TestLogAggregator:
    """Test log aggregation functionality."""
    
    def test_log_entry_creation(self):
        """Test LogEntry creation and serialization."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level='INFO',
            logger='test.logger',
            message='Test message',
            correlation_id='test-correlation',
            user_id='test-user',
            request_id='test-request',
            component='test_component',
            operation='test_operation',
            module='test_module',
            function='test_function',
            line=42,
            process_id=1234,
            thread_id=5678
        )
        
        # Test dictionary conversion
        entry_dict = entry.to_dict()
        assert entry_dict['level'] == 'INFO'
        assert entry_dict['message'] == 'Test message'
        
        # Test JSON conversion
        entry_json = entry.to_json()
        parsed = json.loads(entry_json)
        assert parsed['level'] == 'INFO'
        assert parsed['correlation_id'] == 'test-correlation'
    
    def test_log_batch(self):
        """Test log batching functionality."""
        batch = LogBatch(max_size=3)
        
        # Add entries
        for i in range(2):
            entry = LogEntry(
                timestamp=datetime.utcnow().isoformat(),
                level='INFO',
                logger='test.logger',
                message=f'Test message {i}',
                correlation_id='test-correlation',
                user_id='test-user',
                request_id='test-request',
                component='test_component',
                operation='test_operation',
                module='test_module',
                function='test_function',
                line=42,
                process_id=1234,
                thread_id=5678
            )
            is_full = batch.add(entry)
            assert is_full is False
        
        # Add third entry (should make batch full)
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level='INFO',
            logger='test.logger',
            message='Test message 2',
            correlation_id='test-correlation',
            user_id='test-user',
            request_id='test-request',
            component='test_component',
            operation='test_operation',
            module='test_module',
            function='test_function',
            line=42,
            process_id=1234,
            thread_id=5678
        )
        is_full = batch.add(entry)
        assert is_full is True
        
        # Test JSON conversion
        json_data = batch.to_json()
        parsed = json.loads(json_data)
        assert len(parsed) == 3
    
    def test_log_filter(self):
        """Test log filtering functionality."""
        log_filter = AggregatorLogFilter()
        
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level='DEBUG',
            logger='test.logger',
            message='Debug message',
            correlation_id='test-correlation',
            user_id='test-user',
            request_id='test-request',
            component='test_component',
            operation='test_operation',
            module='test_module',
            function='test_function',
            line=42,
            process_id=1234,
            thread_id=5678
        )
        
        # Should not forward DEBUG when min level is INFO
        should_forward = log_filter.should_forward(entry, LogLevel.INFO)
        assert should_forward is False
        
        # Should forward INFO when min level is INFO
        entry.level = 'INFO'
        should_forward = log_filter.should_forward(entry, LogLevel.INFO)
        assert should_forward is True
        
        # Test with filters
        filters = ['component:test_component']
        should_forward = log_filter.should_forward(entry, LogLevel.INFO, filters)
        assert should_forward is True
        
        filters = ['component:!test_component']
        should_forward = log_filter.should_forward(entry, LogLevel.INFO, filters)
        assert should_forward is False
    
    @pytest.mark.asyncio
    async def test_log_forwarder_file(self):
        """Test log forwarder with file destination."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'forwarded.log')
            
            config = LogDestinationConfig(
                type=LogDestination.FILE,
                url=log_file,
                batch_size=2,
                flush_interval=1
            )
            
            forwarder = LogForwarder(config)
            await forwarder.start()
            
            # Create batch with entries
            batch = LogBatch(max_size=2)
            for i in range(2):
                entry = LogEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    level='INFO',
                    logger='test.logger',
                    message=f'Test message {i}',
                    correlation_id='test-correlation',
                    user_id='test-user',
                    request_id='test-request',
                    component='test_component',
                    operation='test_operation',
                    module='test_module',
                    function='test_function',
                    line=42,
                    process_id=1234,
                    thread_id=5678
                )
                batch.add(entry)
            
            # Forward batch
            success = await forwarder.forward_batch(batch)
            assert success is True
            
            # Verify file content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert 'Test message 0' in content
                assert 'Test message 1' in content
            
            await forwarder.stop()
    
    @pytest.mark.asyncio
    async def test_log_aggregator(self):
        """Test log aggregator functionality."""
        aggregator = LogAggregator()
        
        # Add file destination
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'aggregated.log')
            
            config = LogDestinationConfig(
                type=LogDestination.FILE,
                url=log_file,
                batch_size=2,
                flush_interval=1
            )
            
            aggregator.add_destination('test_file', config)
            await aggregator.start()
            
            # Process some logs
            for i in range(3):
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'INFO',
                    'logger': 'test.logger',
                    'message': f'Test message {i}',
                    'correlation_id': 'test-correlation',
                    'user_id': 'test-user',
                    'request_id': 'test-request',
                    'component': 'test_component',
                    'operation': 'test_operation',
                    'module': 'test_module',
                    'function': 'test_function',
                    'line': 42,
                    'process_id': 1234,
                    'thread_id': 5678
                }
                await aggregator.process_log(log_data)
            
            # Wait for flush
            await asyncio.sleep(2)
            
            # Stop aggregator (should flush remaining logs)
            await aggregator.stop()
            
            # Verify logs were written
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                assert len(lines) >= 3
                
                # Verify JSON format
                for line in lines:
                    if line.strip():
                        parsed = json.loads(line)
                        assert 'message' in parsed
                        assert 'correlation_id' in parsed


class TestLogManager:
    """Test log management functionality."""
    
    def test_log_level_manager(self):
        """Test log level management."""
        level_manager = LogLevelManager()
        
        # Test setting and getting levels
        level_manager.set_level('test.component', 'DEBUG')
        assert level_manager.get_level('test.component') == 'DEBUG'
        
        # Test default level
        assert level_manager.get_level('unknown.component') == 'INFO'
        
        # Test should_log
        assert level_manager.should_log('test.component', 'DEBUG') is True
        assert level_manager.should_log('test.component', 'INFO') is True
        
        level_manager.set_level('test.component', 'ERROR')
        assert level_manager.should_log('test.component', 'DEBUG') is False
        assert level_manager.should_log('test.component', 'ERROR') is True
        
        # Test reset
        level_manager.reset_level('test.component')
        assert level_manager.get_level('test.component') == 'INFO'
    
    def test_filter_condition(self):
        """Test filter conditions."""
        # Test equals condition
        condition = FilterCondition('level', FilterOperator.EQUALS, 'ERROR')
        log_data = {'level': 'ERROR', 'message': 'Test error'}
        assert condition.matches(log_data) is True
        
        log_data = {'level': 'INFO', 'message': 'Test info'}
        assert condition.matches(log_data) is False
        
        # Test contains condition
        condition = FilterCondition('message', FilterOperator.CONTAINS, 'error')
        log_data = {'level': 'INFO', 'message': 'This is an error message'}
        assert condition.matches(log_data) is True
        
        # Test regex condition
        condition = FilterCondition('message', FilterOperator.REGEX, r'\\d+')
        log_data = {'level': 'INFO', 'message': 'Error code 404'}
        assert condition.matches(log_data) is True
        
        log_data = {'level': 'INFO', 'message': 'No numbers here'}
        assert condition.matches(log_data) is False
        
        # Test nested field access
        condition = FilterCondition('extra.user_id', FilterOperator.EQUALS, 'test-user')
        log_data = {'level': 'INFO', 'extra': {'user_id': 'test-user'}}
        assert condition.matches(log_data) is True
    
    def test_log_filter(self):
        """Test log filtering."""
        # Create filter to deny ERROR logs
        conditions = [FilterCondition('level', FilterOperator.EQUALS, 'ERROR')]
        log_filter = LogFilter(
            name='deny_errors',
            description='Deny error logs',
            conditions=conditions,
            action=FilterAction.DENY,
            action_params={}
        )
        
        # Test matching log (should be denied)
        log_data = {'level': 'ERROR', 'message': 'Test error'}
        result = log_filter.apply(log_data)
        assert result is None
        
        # Test non-matching log (should pass through)
        log_data = {'level': 'INFO', 'message': 'Test info'}
        result = log_filter.apply(log_data)
        assert result == log_data
        
        # Test modification filter
        conditions = [FilterCondition('component', FilterOperator.EQUALS, 'test')]
        log_filter = LogFilter(
            name='add_tags',
            description='Add tags to test component logs',
            conditions=conditions,
            action=FilterAction.MODIFY,
            action_params={
                'modifications': {
                    '_add_service': 'test_service',
                    'level': 'DEBUG'  # Modify existing field
                }
            }
        )
        
        log_data = {'level': 'INFO', 'component': 'test', 'message': 'Test message'}
        result = log_filter.apply(log_data)
        
        assert result['service'] == 'test_service'
        assert result['level'] == 'DEBUG'
    
    def test_filter_manager(self):
        """Test filter manager."""
        filter_manager = LogFilterManager()
        
        # Add filter
        conditions = [FilterCondition('level', FilterOperator.EQUALS, 'DEBUG')]
        log_filter = LogFilter(
            name='debug_filter',
            description='Filter debug logs',
            conditions=conditions,
            action=FilterAction.DENY,
            action_params={},
            priority=100
        )
        
        filter_manager.add_filter(log_filter)
        
        # Test filtering
        log_data = {'level': 'DEBUG', 'message': 'Debug message'}
        result = filter_manager.apply_filters(log_data)
        assert result is None
        
        log_data = {'level': 'INFO', 'message': 'Info message'}
        result = filter_manager.apply_filters(log_data)
        assert result == log_data
        
        # Test disable filter
        filter_manager.disable_filter('debug_filter')
        log_data = {'level': 'DEBUG', 'message': 'Debug message'}
        result = filter_manager.apply_filters(log_data)
        assert result == log_data
    
    def test_log_analyzer(self):
        """Test log analyzer."""
        analyzer = LogAnalyzer(window_size=100)
        
        # Add some logs
        for i in range(10):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'INFO' if i % 3 != 0 else 'ERROR',
                'logger': 'test.logger',
                'message': f'Test message {i}',
                'correlation_id': f'corr-{i // 3}',
                'user_id': 'test-user',
                'component': 'test_component' if i % 2 == 0 else 'other_component',
                'operation': f'operation_{i % 3}'
            }
            analyzer.add_log(log_data)
        
        # Analyze patterns
        patterns = analyzer.analyze_patterns()
        
        assert patterns['total_logs'] == 10
        assert 'level_distribution' in patterns
        assert patterns['level_distribution']['INFO'] > 0
        assert patterns['level_distribution']['ERROR'] > 0
        
        assert 'component_activity' in patterns
        assert 'test_component' in patterns['component_activity']
        assert 'other_component' in patterns['component_activity']
        
        assert 'error_patterns' in patterns
        assert patterns['error_patterns']['count'] > 0
        
        assert 'correlation_patterns' in patterns
        
        # Test insights
        insights = analyzer.get_insights()
        assert 'recommendations' in insights
        assert 'alerts' in insights
        assert 'summary' in insights
    
    def test_log_manager_integration(self):
        """Test integrated log manager functionality."""
        log_manager = LogManager()
        
        # Process some logs
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO',
            'logger': 'test.logger',
            'message': 'Test message',
            'correlation_id': 'test-correlation',
            'user_id': 'test-user',
            'component': 'test_component',
            'operation': 'test_operation'
        }
        
        result = log_manager.process_log(log_data)
        assert result is not None
        
        # Get comprehensive stats
        stats = log_manager.get_comprehensive_stats()
        assert 'log_levels' in stats
        assert 'filter_stats' in stats
        assert 'analysis' in stats
        assert 'insights' in stats


@pytest.mark.asyncio
async def test_end_to_end_logging():
    """Test end-to-end logging functionality."""
    # Setup logging
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'e2e_test.log')
        
        LoggingConfig.setup_logging(
            level=logging.INFO,
            format_type='json',
            log_file=log_file,
            console_output=False,
            correlation_tracking=True
        )
        
        # Create logger
        logger = get_logger('e2e.test', 'e2e_component')
        
        # Log with correlation context
        with CorrelationContext('e2e-correlation', 'e2e-user', 'e2e-request'):
            logger.info('Starting E2E test', operation='e2e_test', test_id='123')
            logger.warning('Test warning', operation='e2e_test', warning_type='test')
            logger.error('Test error', operation='e2e_test', error_code='E001')
        
        # Verify log file
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 3
            
            for line in lines:
                if line.strip():
                    log_entry = json.loads(line)
                    assert log_entry['correlation_id'] == 'e2e-correlation'
                    assert log_entry['user_id'] == 'e2e-user'
                    assert log_entry['request_id'] == 'e2e-request'
                    assert log_entry['component'] == 'e2e_component'
                    assert log_entry['operation'] == 'e2e_test'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])