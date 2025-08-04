"""
Unit tests for Cloudflare R2 storage client.
Tests the Spinal Cord fallback system functionality.
"""
import pytest
import json
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from src.spinal_cord.r2_client import (
    CloudflareR2Client, SpinalCordStorage, FallbackTask, R2Object,
    R2StorageError, R2AuthenticationError, R2NotFoundError
)
from src.shared.config import CloudflareSettings


@pytest.fixture
def cloudflare_settings():
    """Mock Cloudflare settings."""
    return CloudflareSettings(
        cloudflare_api_token="test-token",
        cloudflare_account_id="test-account-id",
        cloudflare_r2_bucket="test-bucket"
    )


@pytest.fixture
def sample_fallback_task():
    """Sample fallback task for testing."""
    return FallbackTask(
        id=str(uuid.uuid4()),
        task_type="scrape_url",
        payload={"url": "https://example.com", "priority": True},
        priority=1,
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_session():
    """Mock aiohttp session."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.headers = {}
    return session


class TestFallbackTask:
    """Test FallbackTask data class."""
    
    def test_to_dict(self, sample_fallback_task):
        """Test converting task to dictionary."""
        task_dict = sample_fallback_task.to_dict()
        
        assert task_dict["id"] == sample_fallback_task.id
        assert task_dict["task_type"] == sample_fallback_task.task_type
        assert task_dict["payload"] == sample_fallback_task.payload
        assert task_dict["priority"] == sample_fallback_task.priority
        assert isinstance(task_dict["created_at"], str)  # Should be ISO string
    
    def test_from_dict(self, sample_fallback_task):
        """Test creating task from dictionary."""
        task_dict = sample_fallback_task.to_dict()
        restored_task = FallbackTask.from_dict(task_dict)
        
        assert restored_task.id == sample_fallback_task.id
        assert restored_task.task_type == sample_fallback_task.task_type
        assert restored_task.payload == sample_fallback_task.payload
        assert restored_task.priority == sample_fallback_task.priority
        assert isinstance(restored_task.created_at, datetime)
    
    def test_roundtrip_serialization(self, sample_fallback_task):
        """Test roundtrip serialization (to_dict -> from_dict)."""
        task_dict = sample_fallback_task.to_dict()
        restored_task = FallbackTask.from_dict(task_dict)
        
        # Compare all fields except datetime precision
        assert restored_task.id == sample_fallback_task.id
        assert restored_task.task_type == sample_fallback_task.task_type
        assert restored_task.payload == sample_fallback_task.payload
        assert restored_task.priority == sample_fallback_task.priority
        assert restored_task.retry_count == sample_fallback_task.retry_count


class TestCloudflareR2Client:
    """Test CloudflareR2Client functionality."""
    
    def test_init_with_settings(self, cloudflare_settings):
        """Test client initialization with settings."""
        client = CloudflareR2Client(cloudflare_settings)
        
        assert client.settings == cloudflare_settings
        assert client.session is None
        assert "test-account-id" in client._base_url
    
    def test_init_without_token(self):
        """Test client initialization without API token."""
        settings = CloudflareSettings(
            cloudflare_account_id="test-account-id",
            cloudflare_r2_bucket="test-bucket"
        )
        
        with pytest.raises(ValueError, match="API token is required"):
            CloudflareR2Client(settings)
    
    def test_init_without_account_id(self):
        """Test client initialization without account ID."""
        settings = CloudflareSettings(
            cloudflare_api_token="test-token",
            cloudflare_r2_bucket="test-bucket"
        )
        
        with pytest.raises(ValueError, match="account ID is required"):
            CloudflareR2Client(settings)
    
    @pytest.mark.asyncio
    async def test_connect(self, cloudflare_settings):
        """Test client connection."""
        client = CloudflareR2Client(cloudflare_settings)
        
        await client.connect()
        
        assert client.session is not None
        assert isinstance(client.session, aiohttp.ClientSession)
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_close(self, cloudflare_settings):
        """Test client disconnection."""
        client = CloudflareR2Client(cloudflare_settings)
        
        await client.connect()
        assert client.session is not None
        
        await client.close()
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self, cloudflare_settings):
        """Test client as async context manager."""
        async with CloudflareR2Client(cloudflare_settings) as client:
            assert client.session is not None
        
        # Session should be closed after context exit
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_put_object_dict(self, cloudflare_settings, mock_session):
        """Test storing a dictionary object."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        test_data = {"key": "value", "number": 42}
        result = await client.put_object("test-key", test_data)
        
        assert result is True
        mock_session.request.assert_called_once()
        
        # Verify the call was made with JSON data
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "PUT"
        assert "test-key" in call_args[1]["url"]
        assert isinstance(call_args[1]["data"], bytes)
    
    @pytest.mark.asyncio
    async def test_put_object_string(self, cloudflare_settings, mock_session):
        """Test storing a string object."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.put_object("test-key", "test string data")
        
        assert result is True
        mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_put_object_bytes(self, cloudflare_settings, mock_session):
        """Test storing bytes object."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        test_data = b"binary data"
        result = await client.put_object("test-key", test_data)
        
        assert result is True
        mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_object_success(self, cloudflare_settings, mock_session):
        """Test successful object retrieval."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful response
        test_data = b"test data"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = test_data
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.get_object("test-key")
        
        assert result == test_data
        mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_object_not_found(self, cloudflare_settings, mock_session):
        """Test object retrieval when not found."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.get_object("nonexistent-key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_object_json(self, cloudflare_settings, mock_session):
        """Test JSON object retrieval."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful response with JSON data
        test_data = {"key": "value", "number": 42}
        json_data = json.dumps(test_data).encode('utf-8')
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = json_data
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.get_object_json("test-key")
        
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_get_object_json_invalid(self, cloudflare_settings, mock_session):
        """Test JSON object retrieval with invalid JSON."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock response with invalid JSON
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"invalid json {"
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(R2StorageError, match="Invalid JSON"):
            await client.get_object_json("test-key")
    
    @pytest.mark.asyncio
    async def test_delete_object_success(self, cloudflare_settings, mock_session):
        """Test successful object deletion."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 204
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.delete_object("test-key")
        
        assert result is True
        mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_object_not_found(self, cloudflare_settings, mock_session):
        """Test object deletion when not found."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.delete_object("nonexistent-key")
        
        assert result is True  # 404 is considered success for deletion
    
    @pytest.mark.asyncio
    async def test_object_exists_true(self, cloudflare_settings, mock_session):
        """Test object existence check when object exists."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock successful HEAD response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.object_exists("test-key")
        
        assert result is True
        
        # Verify HEAD request was made
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "HEAD"
    
    @pytest.mark.asyncio
    async def test_object_exists_false(self, cloudflare_settings, mock_session):
        """Test object existence check when object doesn't exist."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.object_exists("nonexistent-key")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, cloudflare_settings, mock_session):
        """Test authentication error handling."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock 401 response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(R2AuthenticationError):
            await client.get_object("test-key")
    
    @pytest.mark.asyncio
    async def test_server_error(self, cloudflare_settings, mock_session):
        """Test server error handling."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock 500 response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(R2StorageError, match="R2 API error 500"):
            await client.get_object("test-key")
    
    def test_parse_list_objects_response(self, cloudflare_settings):
        """Test XML parsing for list objects response."""
        client = CloudflareR2Client(cloudflare_settings)
        
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
        <ListBucketResult>
            <Contents>
                <Key>test-key-1</Key>
                <Size>1024</Size>
                <LastModified>2023-01-01T12:00:00.000Z</LastModified>
                <ETag>"abc123"</ETag>
            </Contents>
            <Contents>
                <Key>test-key-2</Key>
                <Size>2048</Size>
                <LastModified>2023-01-02T12:00:00.000Z</LastModified>
                <ETag>"def456"</ETag>
            </Contents>
        </ListBucketResult>"""
        
        objects = client._parse_list_objects_response(xml_response)
        
        assert len(objects) == 2
        assert objects[0].key == "test-key-1"
        assert objects[0].size == 1024
        assert objects[0].etag == "abc123"
        assert objects[1].key == "test-key-2"
        assert objects[1].size == 2048


class TestSpinalCordStorage:
    """Test SpinalCordStorage high-level interface."""
    
    @pytest.fixture
    def mock_r2_client(self):
        """Mock R2 client."""
        return AsyncMock(spec=CloudflareR2Client)
    
    @pytest.mark.asyncio
    async def test_store_task(self, mock_r2_client, sample_fallback_task):
        """Test storing a single task."""
        storage = SpinalCordStorage(mock_r2_client)
        mock_r2_client.put_object.return_value = True
        
        key = await storage.store_task(sample_fallback_task)
        
        assert key.startswith("tasks/")
        assert key.endswith(".json")
        mock_r2_client.put_object.assert_called_once()
        
        # Verify the task data was serialized correctly
        call_args = mock_r2_client.put_object.call_args
        task_data = call_args[0][1]  # Second argument is the data
        assert task_data["id"] == sample_fallback_task.id
        assert task_data["task_type"] == sample_fallback_task.task_type
    
    @pytest.mark.asyncio
    async def test_retrieve_task(self, mock_r2_client, sample_fallback_task):
        """Test retrieving a single task."""
        storage = SpinalCordStorage(mock_r2_client)
        
        # Mock the R2 client to return task data
        task_data = sample_fallback_task.to_dict()
        mock_r2_client.get_object_json.return_value = task_data
        
        retrieved_task = await storage.retrieve_task(sample_fallback_task.id)
        
        assert retrieved_task is not None
        assert retrieved_task.id == sample_fallback_task.id
        assert retrieved_task.task_type == sample_fallback_task.task_type
        assert retrieved_task.payload == sample_fallback_task.payload
        
        mock_r2_client.get_object_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_task_not_found(self, mock_r2_client):
        """Test retrieving a task that doesn't exist."""
        storage = SpinalCordStorage(mock_r2_client)
        mock_r2_client.get_object_json.return_value = None
        
        retrieved_task = await storage.retrieve_task("nonexistent-id")
        
        assert retrieved_task is None
    
    @pytest.mark.asyncio
    async def test_delete_task(self, mock_r2_client):
        """Test deleting a task."""
        storage = SpinalCordStorage(mock_r2_client)
        mock_r2_client.delete_object.return_value = True
        
        result = await storage.delete_task("test-task-id")
        
        assert result is True
        mock_r2_client.delete_object.assert_called_once()
        
        # Verify correct key was used
        call_args = mock_r2_client.delete_object.call_args
        key = call_args[0][0]
        assert key == "tasks/test-task-id.json"
    
    @pytest.mark.asyncio
    async def test_store_task_batch(self, mock_r2_client, sample_fallback_task):
        """Test storing a batch of tasks."""
        storage = SpinalCordStorage(mock_r2_client)
        mock_r2_client.put_object.return_value = True
        
        tasks = [sample_fallback_task]
        key = await storage.store_task_batch(tasks, "test-batch")
        
        assert key.startswith("batches/")
        assert key.endswith(".json")
        mock_r2_client.put_object.assert_called_once()
        
        # Verify batch data structure
        call_args = mock_r2_client.put_object.call_args
        batch_data = call_args[0][1]  # Second argument is the data
        assert batch_data["task_count"] == 1
        assert len(batch_data["tasks"]) == 1
        assert batch_data["tasks"][0]["id"] == sample_fallback_task.id
    
    @pytest.mark.asyncio
    async def test_retrieve_task_batch(self, mock_r2_client, sample_fallback_task):
        """Test retrieving a batch of tasks."""
        storage = SpinalCordStorage(mock_r2_client)
        
        # Mock batch data
        batch_data = {
            "batch_id": "test-batch",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_count": 1,
            "tasks": [sample_fallback_task.to_dict()]
        }
        mock_r2_client.get_object_json.return_value = batch_data
        
        retrieved_tasks = await storage.retrieve_task_batch("test-batch")
        
        assert retrieved_tasks is not None
        assert len(retrieved_tasks) == 1
        assert retrieved_tasks[0].id == sample_fallback_task.id
        
        mock_r2_client.get_object_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_stored_tasks(self, mock_r2_client):
        """Test listing stored task IDs."""
        storage = SpinalCordStorage(mock_r2_client)
        
        # Mock R2 objects
        mock_objects = [
            R2Object(
                key="tasks/task-1.json",
                size=1024,
                last_modified=datetime.now(timezone.utc),
                etag="abc123"
            ),
            R2Object(
                key="tasks/task-2.json",
                size=2048,
                last_modified=datetime.now(timezone.utc),
                etag="def456"
            )
        ]
        mock_r2_client.list_objects.return_value = mock_objects
        
        task_ids = await storage.list_stored_tasks()
        
        assert len(task_ids) == 2
        assert "task-1" in task_ids
        assert "task-2" in task_ids
        
        mock_r2_client.list_objects.assert_called_once_with(prefix="tasks/", max_keys=100)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_tasks(self, mock_r2_client):
        """Test cleaning up old tasks."""
        storage = SpinalCordStorage(mock_r2_client)
        
        # Mock old and new objects
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)  # Older than 24 hours
        new_time = datetime.now(timezone.utc) - timedelta(hours=1)   # Recent
        
        mock_objects = [
            R2Object(key="tasks/old-task.json", size=1024, last_modified=old_time, etag="abc"),
            R2Object(key="tasks/new-task.json", size=1024, last_modified=new_time, etag="def")
        ]
        mock_r2_client.list_objects.return_value = mock_objects
        mock_r2_client.delete_object.return_value = True
        
        cleaned_count = await storage.cleanup_old_tasks(max_age_hours=24)
        
        assert cleaned_count == 1
        mock_r2_client.delete_object.assert_called_once_with("tasks/old-task.json")
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_r2_client):
        """Test SpinalCordStorage as context manager."""
        async with SpinalCordStorage(mock_r2_client) as storage:
            assert storage.r2_client == mock_r2_client
        
        mock_r2_client.connect.assert_called_once()
        mock_r2_client.close.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_storage_error_propagation(self, cloudflare_settings, mock_session):
        """Test that storage errors are properly propagated."""
        client = CloudflareR2Client(cloudflare_settings)
        client.session = mock_session
        
        # Mock network error
        mock_session.request.side_effect = aiohttp.ClientError("Network error")
        
        with pytest.raises(R2StorageError, match="Network error"):
            await client.get_object("test-key")
    
    @pytest.mark.asyncio
    async def test_spinal_cord_error_handling(self, sample_fallback_task):
        """Test error handling in SpinalCordStorage."""
        mock_r2_client = AsyncMock(spec=CloudflareR2Client)
        mock_r2_client.put_object.side_effect = R2StorageError("Storage failed")
        
        storage = SpinalCordStorage(mock_r2_client)
        
        with pytest.raises(R2StorageError):
            await storage.store_task(sample_fallback_task)