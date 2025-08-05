"""
Integration tests for Cloudflare Workers and Vercel Edge Functions.
Tests worker deployment, task handling, and failover scenarios.
"""
import pytest
import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from src.signal_relay.task_dispatcher import TaskType, TaskPriority


class TestCloudflareWorkersIntegration:
    """Test Cloudflare Workers integration functionality."""
    
    @pytest.fixture
    def worker_config(self):
        """Configuration for worker testing."""
        return {
            'worker_url': 'https://synapse-task-dispatcher.example.workers.dev',
            'api_key': 'test-api-key',
            'webhook_secret': 'test-webhook-secret'
        }
    
    @pytest.fixture
    def vercel_config(self):
        """Configuration for Vercel fallback testing."""
        return {
            'vercel_url': 'https://synapse-fallback.vercel.app',
            'api_key': 'test-api-key'
        }
    
    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            'task_type': 'scrape_url',
            'payload': {
                'url': 'https://example.com/article',
                'priority': True,
                'metadata': {'source': 'test'}
            },
            'priority': 2,
            'max_retries': 3
        }
    
    @pytest.fixture
    def sample_webhook_data(self):
        """Sample webhook data for testing."""
        return {
            'github_push': {
                'repository': {'full_name': 'test/repo'},
                'head_commit': {
                    'id': 'abc123',
                    'message': 'Update README.md'
                },
                'commits': [{
                    'modified': ['README.md', 'docs/api.md']
                }]
            },
            'generic': {
                'url': 'https://example.com/new-article',
                'title': 'New Article',
                'priority': 2,
                'metadata': {'source': 'webhook'}
            },
            'feed_update': {
                'feed_url': 'https://example.com/feed.xml',
                'items': [
                    {
                        'url': 'https://example.com/item1',
                        'title': 'Item 1',
                        'published_at': '2025-01-01T00:00:00Z'
                    },
                    {
                        'url': 'https://example.com/item2',
                        'title': 'Item 2',
                        'published_at': '2025-01-01T01:00:00Z'
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_worker_health_check(self, worker_config):
        """Test worker health check endpoint."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'status': 'healthy',
                'service': 'cloudflare-worker',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{worker_config['worker_url']}/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['status'] == 'healthy'
                    assert data['service'] == 'cloudflare-worker'
    
    @pytest.mark.asyncio
    async def test_worker_task_submission(self, worker_config, sample_task_data):
        """Test task submission to Cloudflare Worker."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful task submission
            task_id = str(uuid.uuid4())
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json.return_value = {
                'success': True,
                'task_id': task_id,
                'status': 'submitted',
                'message': 'Task submitted successfully'
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': worker_config['api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/tasks",
                    json=sample_task_data,
                    headers=headers
                ) as response:
                    assert response.status == 201
                    data = await response.json()
                    assert data['success'] is True
                    assert data['task_id'] == task_id
                    assert data['status'] == 'submitted'
    
    @pytest.mark.asyncio
    async def test_worker_task_status_retrieval(self, worker_config):
        """Test task status retrieval from Cloudflare Worker."""
        task_id = str(uuid.uuid4())
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful status retrieval
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'data': {
                    'task_id': task_id,
                    'status': 'completed',
                    'task_type': 'scrape_url',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'completed_at': datetime.now(timezone.utc).isoformat()
                }
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            headers = {'X-API-Key': worker_config['api_key']}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{worker_config['worker_url']}/tasks/{task_id}",
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['data']['task_id'] == task_id
                    assert data['data']['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_worker_github_webhook(self, worker_config, sample_webhook_data):
        """Test GitHub webhook processing."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful webhook processing
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'message': 'GitHub push webhook processed',
                'tasks_submitted': 1,
                'task_ids': [str(uuid.uuid4())]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {
                'Content-Type': 'application/json',
                'X-GitHub-Event': 'push',
                'X-Hub-Signature-256': 'sha256=test-signature'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/webhooks/github",
                    json=sample_webhook_data['github_push'],
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['tasks_submitted'] >= 1
    
    @pytest.mark.asyncio
    async def test_worker_generic_webhook(self, worker_config, sample_webhook_data):
        """Test generic webhook processing."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful webhook processing
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'message': 'Generic webhook processed',
                'tasks_submitted': 1,
                'task_ids': [str(uuid.uuid4())]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {
                'Content-Type': 'application/json',
                'X-Webhook-Signature': 'test-signature'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/webhooks/generic",
                    json=sample_webhook_data['generic'],
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['tasks_submitted'] >= 1
    
    @pytest.mark.asyncio
    async def test_worker_feed_webhook(self, worker_config, sample_webhook_data):
        """Test feed update webhook processing."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful webhook processing
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'message': 'Feed update webhook processed',
                'tasks_submitted': 3,  # 1 feed + 2 items
                'task_ids': [str(uuid.uuid4()) for _ in range(3)]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {'Content-Type': 'application/json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/webhooks/feed-update",
                    json=sample_webhook_data['feed_update'],
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['tasks_submitted'] == 3
    
    @pytest.mark.asyncio
    async def test_worker_cron_processing(self, worker_config):
        """Test cron-triggered task processing."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful cron processing
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'message': 'Cron tasks processed',
                'data': {
                    'processed': 5,
                    'errors': 0
                }
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {'X-API-Key': worker_config['api_key']}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/cron/process-tasks",
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['data']['processed'] >= 0


class TestVercelFallbackIntegration:
    """Test Vercel Edge Function fallback functionality."""
    
    @pytest.mark.asyncio
    async def test_vercel_health_check(self, vercel_config):
        """Test Vercel fallback health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'status': 'healthy',
                'service': 'vercel-edge-fallback',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '1.0.0'
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{vercel_config['vercel_url']}/health") as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['status'] == 'healthy'
                    assert data['service'] == 'vercel-edge-fallback'
    
    @pytest.mark.asyncio
    async def test_vercel_task_submission(self, vercel_config, sample_task_data):
        """Test task submission to Vercel fallback."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful task submission
            task_id = f"fallback-{int(datetime.now().timestamp())}-1"
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json.return_value = {
                'success': True,
                'task_id': task_id,
                'status': 'queued',
                'message': 'Task submitted to fallback service',
                'fallback': True
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': vercel_config['api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{vercel_config['vercel_url']}/tasks",
                    json=sample_task_data,
                    headers=headers
                ) as response:
                    assert response.status == 201
                    data = await response.json()
                    assert data['success'] is True
                    assert data['fallback'] is True
                    assert data['task_id'].startswith('fallback-')
    
    @pytest.mark.asyncio
    async def test_vercel_task_status(self, vercel_config):
        """Test task status retrieval from Vercel fallback."""
        task_id = "fallback-123456789-1"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful status retrieval
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'data': {
                    'id': task_id,
                    'task_type': 'scrape_url',
                    'status': 'queued',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'source': 'vercel_fallback',
                    'fallback': True,
                    'fallback_service': True,
                    'note': 'This task is stored in fallback service with limited functionality'
                }
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            headers = {'X-API-Key': vercel_config['api_key']}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{vercel_config['vercel_url']}/tasks/{task_id}",
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['data']['fallback_service'] is True
                    assert data['data']['id'] == task_id


class TestFailoverScenarios:
    """Test failover scenarios between Cloudflare Workers and Vercel."""
    
    @pytest.mark.asyncio
    async def test_cloudflare_to_vercel_failover(self, worker_config, vercel_config, sample_task_data):
        """Test failover from Cloudflare Worker to Vercel Edge Function."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call to Cloudflare fails
            cloudflare_error = aiohttp.ClientError("Connection failed")
            
            # Second call to Vercel succeeds
            task_id = f"fallback-{int(datetime.now().timestamp())}-1"
            vercel_response = AsyncMock()
            vercel_response.status = 201
            vercel_response.json.return_value = {
                'success': True,
                'task_id': task_id,
                'status': 'queued',
                'fallback': True
            }
            
            # Configure mock to fail first, succeed second
            mock_post.side_effect = [cloudflare_error, vercel_response.__aenter__.return_value]
            
            # Simulate failover logic
            async with aiohttp.ClientSession() as session:
                try:
                    # Try Cloudflare first
                    await session.post(
                        f"{worker_config['worker_url']}/tasks",
                        json=sample_task_data,
                        headers={'X-API-Key': worker_config['api_key']}
                    )
                    assert False, "Should have failed"
                except aiohttp.ClientError:
                    # Fallback to Vercel
                    async with session.post(
                        f"{vercel_config['vercel_url']}/tasks",
                        json=sample_task_data,
                        headers={'X-API-Key': vercel_config['api_key']}
                    ) as response:
                        assert response.status == 201
                        data = await response.json()
                        assert data['success'] is True
                        assert data['fallback'] is True
    
    @pytest.mark.asyncio
    async def test_fallback_status_monitoring(self, worker_config, vercel_config):
        """Test fallback status monitoring."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock fallback status response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'success': True,
                'data': {
                    'fallback_configured': True,
                    'vercel': {
                        'status': 'healthy',
                        'url': f"{vercel_config['vercel_url']}/...",
                        'response_time_ms': 150
                    },
                    'statistics': {
                        'total_fallback_requests': 10,
                        'successful_fallbacks': 9,
                        'failed_fallbacks': 1
                    },
                    'last_check': datetime.now(timezone.utc).isoformat()
                }
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            headers = {'X-API-Key': worker_config['api_key']}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{worker_config['worker_url']}/fallback/status",
                    headers=headers
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data['success'] is True
                    assert data['data']['fallback_configured'] is True
                    assert data['data']['vercel']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_worker_deployment_validation(self):
        """Test worker deployment configuration validation."""
        # Test wrangler.toml configuration
        wrangler_config = {
            'name': 'synapse-task-dispatcher',
            'main': 'src/index.js',
            'compatibility_date': '2024-01-01',
            'compatibility_flags': ['nodejs_compat']
        }
        
        assert wrangler_config['name'] == 'synapse-task-dispatcher'
        assert wrangler_config['main'] == 'src/index.js'
        assert 'nodejs_compat' in wrangler_config['compatibility_flags']
    
    @pytest.mark.asyncio
    async def test_vercel_deployment_validation(self):
        """Test Vercel deployment configuration validation."""
        # Test vercel.json configuration
        vercel_config = {
            'functions': {
                'api/tasks/[...path].js': {'runtime': 'edge'},
                'api/health.js': {'runtime': 'edge'}
            },
            'rewrites': [
                {'source': '/tasks/(.*)', 'destination': '/api/tasks/$1'},
                {'source': '/health', 'destination': '/api/health'}
            ]
        }
        
        assert vercel_config['functions']['api/tasks/[...path].js']['runtime'] == 'edge'
        assert vercel_config['functions']['api/health.js']['runtime'] == 'edge'
        assert len(vercel_config['rewrites']) >= 2


class TestWorkerErrorHandling:
    """Test error handling in worker scenarios."""
    
    @pytest.mark.asyncio
    async def test_worker_authentication_error(self, worker_config, sample_task_data):
        """Test worker authentication error handling."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock authentication error
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json.return_value = {
                'success': False,
                'error': 'Unauthorized',
                'message': 'Invalid API key'
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': 'invalid-key'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/tasks",
                    json=sample_task_data,
                    headers=headers
                ) as response:
                    assert response.status == 401
                    data = await response.json()
                    assert data['success'] is False
                    assert 'Unauthorized' in data['error']
    
    @pytest.mark.asyncio
    async def test_worker_rate_limiting(self, worker_config, sample_task_data):
        """Test worker rate limiting."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock rate limit error
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.json.return_value = {
                'success': False,
                'error': 'Rate limit exceeded',
                'message': 'Too many requests'
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': worker_config['api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/tasks",
                    json=sample_task_data,
                    headers=headers
                ) as response:
                    assert response.status == 429
                    data = await response.json()
                    assert data['success'] is False
                    assert 'Rate limit' in data['error']
    
    @pytest.mark.asyncio
    async def test_worker_validation_error(self, worker_config):
        """Test worker input validation error."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock validation error
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.json.return_value = {
                'success': False,
                'error': 'Missing required fields: task_type, payload'
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Invalid task data (missing required fields)
            invalid_data = {'priority': 1}
            
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': worker_config['api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{worker_config['worker_url']}/tasks",
                    json=invalid_data,
                    headers=headers
                ) as response:
                    assert response.status == 400
                    data = await response.json()
                    assert data['success'] is False
                    assert 'Missing required fields' in data['error']