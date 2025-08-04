"""
Spinal Cord - Cloudflare R2 Storage Client
Layer 2: Signal Network

This module implements the Cloudflare R2 storage client for the Spinal Cord fallback system.
Provides resilient task storage when the primary database is unavailable.
"""
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
import structlog
from urllib.parse import urljoin

from ..shared.config import get_settings, CloudflareSettings

logger = structlog.get_logger(__name__)


@dataclass
class R2Object:
    """Represents an object stored in R2."""
    key: str
    size: int
    last_modified: datetime
    etag: str
    content_type: str = "application/json"


@dataclass
class FallbackTask:
    """Represents a task stored in the fallback system."""
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FallbackTask':
        """Create FallbackTask from dictionary."""
        # Convert ISO string back to datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FallbackTask to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        return data


class R2StorageError(Exception):
    """Base exception for R2 storage operations."""
    pass


class R2AuthenticationError(R2StorageError):
    """Authentication error with R2."""
    pass


class R2NotFoundError(R2StorageError):
    """Object not found in R2."""
    pass


class CloudflareR2Client:
    """
    Cloudflare R2 storage client for the Spinal Cord fallback system.
    
    Provides resilient storage for tasks when the primary database is unavailable.
    Uses S3-compatible API for object storage operations.
    """
    
    def __init__(self, settings: Optional[CloudflareSettings] = None):
        self.settings = settings or get_settings().cloudflare
        self.session: Optional[aiohttp.ClientSession] = None
        self._base_url = self._get_base_url()
        
        # Validate required settings
        if not self.settings.cloudflare_api_token:
            raise ValueError("Cloudflare API token is required")
        if not self.settings.cloudflare_account_id:
            raise ValueError("Cloudflare account ID is required")
    
    def _get_base_url(self) -> str:
        """Get the R2 API base URL."""
        account_id = self.settings.cloudflare_account_id
        return f"https://{account_id}.r2.cloudflarestorage.com"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Initialize the HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "Authorization": f"Bearer {self.settings.cloudflare_api_token}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            logger.info("R2 client connected", bucket=self.settings.cloudflare_r2_bucket)
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("R2 client disconnected")
    
    async def _make_request(
        self, 
        method: str, 
        path: str, 
        data: Optional[bytes] = None,
        params: Optional[Dict[str, str]] = None
    ) -> aiohttp.ClientResponse:
        """Make an authenticated request to R2."""
        if not self.session:
            await self.connect()
        
        url = urljoin(self._base_url, f"/{self.settings.cloudflare_r2_bucket}{path}")
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                data=data,
                params=params
            ) as response:
                if response.status == 401:
                    raise R2AuthenticationError("Invalid API token or insufficient permissions")
                elif response.status == 404:
                    raise R2NotFoundError(f"Object not found: {path}")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise R2StorageError(f"R2 API error {response.status}: {error_text}")
                
                return response
                
        except aiohttp.ClientError as e:
            logger.error("R2 client error", error=str(e), method=method, path=path)
            raise R2StorageError(f"Network error: {str(e)}")
    
    async def put_object(
        self, 
        key: str, 
        data: Union[str, bytes, Dict[str, Any]], 
        content_type: str = "application/json"
    ) -> bool:
        """
        Store an object in R2.
        
        Args:
            key: Object key (path)
            data: Data to store (string, bytes, or dict for JSON)
            content_type: Content type of the object
            
        Returns:
            True if successful
        """
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data, default=str).encode('utf-8')
                content_type = "application/json"
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Add content-type header for this request
            headers = {"Content-Type": content_type}
            
            # Make request with custom headers
            if self.session:
                self.session.headers.update(headers)
            
            response = await self._make_request("PUT", f"/{key}", data=data_bytes)
            
            success = response.status in (200, 201)
            if success:
                logger.info("Object stored in R2", key=key, size=len(data_bytes))
            else:
                logger.error("Failed to store object in R2", key=key, status=response.status)
            
            return success
            
        except Exception as e:
            logger.error("Error storing object in R2", key=key, error=str(e))
            raise R2StorageError(f"Failed to store object {key}: {str(e)}")
    
    async def get_object(self, key: str) -> Optional[bytes]:
        """
        Retrieve an object from R2.
        
        Args:
            key: Object key (path)
            
        Returns:
            Object data as bytes, or None if not found
        """
        try:
            response = await self._make_request("GET", f"/{key}")
            
            if response.status == 200:
                data = await response.read()
                logger.info("Object retrieved from R2", key=key, size=len(data))
                return data
            else:
                logger.warning("Object not found in R2", key=key, status=response.status)
                return None
                
        except R2NotFoundError:
            logger.info("Object not found in R2", key=key)
            return None
        except Exception as e:
            logger.error("Error retrieving object from R2", key=key, error=str(e))
            raise R2StorageError(f"Failed to retrieve object {key}: {str(e)}")
    
    async def get_object_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a JSON object from R2.
        
        Args:
            key: Object key (path)
            
        Returns:
            Parsed JSON data, or None if not found
        """
        data = await self.get_object(key)
        if data:
            try:
                return json.loads(data.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON from R2 object", key=key, error=str(e))
                raise R2StorageError(f"Invalid JSON in object {key}: {str(e)}")
        return None
    
    async def delete_object(self, key: str) -> bool:
        """
        Delete an object from R2.
        
        Args:
            key: Object key (path)
            
        Returns:
            True if successful or object didn't exist
        """
        try:
            response = await self._make_request("DELETE", f"/{key}")
            
            success = response.status in (200, 204, 404)  # 404 means already deleted
            if success:
                logger.info("Object deleted from R2", key=key)
            else:
                logger.error("Failed to delete object from R2", key=key, status=response.status)
            
            return success
            
        except R2NotFoundError:
            # Object already doesn't exist
            logger.info("Object already deleted from R2", key=key)
            return True
        except Exception as e:
            logger.error("Error deleting object from R2", key=key, error=str(e))
            raise R2StorageError(f"Failed to delete object {key}: {str(e)}")
    
    async def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[R2Object]:
        """
        List objects in R2 bucket.
        
        Args:
            prefix: Key prefix to filter objects
            max_keys: Maximum number of objects to return
            
        Returns:
            List of R2Object instances
        """
        try:
            params = {"max-keys": str(max_keys)}
            if prefix:
                params["prefix"] = prefix
            
            response = await self._make_request("GET", "", params=params)
            
            if response.status == 200:
                # Parse XML response (S3-compatible)
                text = await response.text()
                objects = self._parse_list_objects_response(text)
                logger.info("Listed objects from R2", prefix=prefix, count=len(objects))
                return objects
            else:
                logger.error("Failed to list objects from R2", status=response.status)
                return []
                
        except Exception as e:
            logger.error("Error listing objects from R2", prefix=prefix, error=str(e))
            raise R2StorageError(f"Failed to list objects: {str(e)}")
    
    def _parse_list_objects_response(self, xml_text: str) -> List[R2Object]:
        """Parse XML response from list objects API."""
        # Simple XML parsing for S3 ListObjects response
        # In production, you might want to use xml.etree.ElementTree
        objects = []
        
        # This is a simplified parser - in production use proper XML parsing
        import re
        
        # Find all <Contents> blocks
        contents_pattern = r'<Contents>(.*?)</Contents>'
        contents_matches = re.findall(contents_pattern, xml_text, re.DOTALL)
        
        for content in contents_matches:
            # Extract key
            key_match = re.search(r'<Key>(.*?)</Key>', content)
            if not key_match:
                continue
            key = key_match.group(1)
            
            # Extract size
            size_match = re.search(r'<Size>(\d+)</Size>', content)
            size = int(size_match.group(1)) if size_match else 0
            
            # Extract last modified
            modified_match = re.search(r'<LastModified>(.*?)</LastModified>', content)
            last_modified = datetime.now(timezone.utc)
            if modified_match:
                try:
                    last_modified = datetime.fromisoformat(modified_match.group(1).replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            # Extract ETag
            etag_match = re.search(r'<ETag>"?(.*?)"?</ETag>', content)
            etag = etag_match.group(1) if etag_match else ""
            
            objects.append(R2Object(
                key=key,
                size=size,
                last_modified=last_modified,
                etag=etag
            ))
        
        return objects
    
    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists in R2.
        
        Args:
            key: Object key (path)
            
        Returns:
            True if object exists
        """
        try:
            response = await self._make_request("HEAD", f"/{key}")
            exists = response.status == 200
            logger.debug("Checked object existence in R2", key=key, exists=exists)
            return exists
            
        except R2NotFoundError:
            return False
        except Exception as e:
            logger.error("Error checking object existence in R2", key=key, error=str(e))
            return False
    
    async def get_object_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get object metadata from R2.
        
        Args:
            key: Object key (path)
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            response = await self._make_request("HEAD", f"/{key}")
            
            if response.status == 200:
                metadata = {
                    "content_length": response.headers.get("Content-Length"),
                    "content_type": response.headers.get("Content-Type"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "etag": response.headers.get("ETag")
                }
                logger.debug("Retrieved object metadata from R2", key=key)
                return metadata
            else:
                return None
                
        except R2NotFoundError:
            return None
        except Exception as e:
            logger.error("Error getting object metadata from R2", key=key, error=str(e))
            return None


class SpinalCordStorage:
    """
    High-level interface for Spinal Cord fallback storage.
    
    Provides task-specific operations using the R2 client.
    """
    
    def __init__(self, r2_client: Optional[CloudflareR2Client] = None):
        self.r2_client = r2_client or CloudflareR2Client()
        self.task_prefix = "tasks/"
        self.batch_prefix = "batches/"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.r2_client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.r2_client.close()
    
    def _generate_task_key(self, task_id: Optional[str] = None) -> str:
        """Generate a unique key for a task."""
        if task_id is None:
            task_id = str(uuid.uuid4())
        return f"{self.task_prefix}{task_id}.json"
    
    def _generate_batch_key(self, batch_id: Optional[str] = None) -> str:
        """Generate a unique key for a batch of tasks."""
        if batch_id is None:
            batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return f"{self.batch_prefix}{batch_id}.json"
    
    async def store_task(self, task: FallbackTask) -> str:
        """
        Store a single task in the fallback storage.
        
        Args:
            task: FallbackTask to store
            
        Returns:
            Storage key for the task
        """
        try:
            key = self._generate_task_key(task.id)
            task_data = task.to_dict()
            
            success = await self.r2_client.put_object(key, task_data)
            if success:
                logger.info("Task stored in Spinal Cord", task_id=task.id, key=key)
                return key
            else:
                raise R2StorageError(f"Failed to store task {task.id}")
                
        except Exception as e:
            logger.error("Error storing task in Spinal Cord", task_id=task.id, error=str(e))
            raise
    
    async def retrieve_task(self, task_id: str) -> Optional[FallbackTask]:
        """
        Retrieve a task from fallback storage.
        
        Args:
            task_id: Task ID to retrieve
            
        Returns:
            FallbackTask instance or None if not found
        """
        try:
            key = self._generate_task_key(task_id)
            task_data = await self.r2_client.get_object_json(key)
            
            if task_data:
                task = FallbackTask.from_dict(task_data)
                logger.info("Task retrieved from Spinal Cord", task_id=task_id)
                return task
            else:
                logger.info("Task not found in Spinal Cord", task_id=task_id)
                return None
                
        except Exception as e:
            logger.error("Error retrieving task from Spinal Cord", task_id=task_id, error=str(e))
            raise
    
    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from fallback storage.
        
        Args:
            task_id: Task ID to delete
            
        Returns:
            True if successful
        """
        try:
            key = self._generate_task_key(task_id)
            success = await self.r2_client.delete_object(key)
            
            if success:
                logger.info("Task deleted from Spinal Cord", task_id=task_id)
            
            return success
            
        except Exception as e:
            logger.error("Error deleting task from Spinal Cord", task_id=task_id, error=str(e))
            raise
    
    async def store_task_batch(self, tasks: List[FallbackTask], batch_id: Optional[str] = None) -> str:
        """
        Store a batch of tasks in fallback storage.
        
        Args:
            tasks: List of FallbackTask instances
            batch_id: Optional batch ID
            
        Returns:
            Storage key for the batch
        """
        try:
            key = self._generate_batch_key(batch_id)
            batch_data = {
                "batch_id": batch_id or key.split('/')[-1].replace('.json', ''),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "task_count": len(tasks),
                "tasks": [task.to_dict() for task in tasks]
            }
            
            success = await self.r2_client.put_object(key, batch_data)
            if success:
                logger.info("Task batch stored in Spinal Cord", batch_id=batch_id, task_count=len(tasks), key=key)
                return key
            else:
                raise R2StorageError(f"Failed to store task batch {batch_id}")
                
        except Exception as e:
            logger.error("Error storing task batch in Spinal Cord", batch_id=batch_id, error=str(e))
            raise
    
    async def retrieve_task_batch(self, batch_id: str) -> Optional[List[FallbackTask]]:
        """
        Retrieve a batch of tasks from fallback storage.
        
        Args:
            batch_id: Batch ID to retrieve
            
        Returns:
            List of FallbackTask instances or None if not found
        """
        try:
            key = self._generate_batch_key(batch_id)
            batch_data = await self.r2_client.get_object_json(key)
            
            if batch_data and "tasks" in batch_data:
                tasks = [FallbackTask.from_dict(task_data) for task_data in batch_data["tasks"]]
                logger.info("Task batch retrieved from Spinal Cord", batch_id=batch_id, task_count=len(tasks))
                return tasks
            else:
                logger.info("Task batch not found in Spinal Cord", batch_id=batch_id)
                return None
                
        except Exception as e:
            logger.error("Error retrieving task batch from Spinal Cord", batch_id=batch_id, error=str(e))
            raise
    
    async def list_stored_tasks(self, limit: int = 100) -> List[str]:
        """
        List all stored task IDs.
        
        Args:
            limit: Maximum number of task IDs to return
            
        Returns:
            List of task IDs
        """
        try:
            objects = await self.r2_client.list_objects(prefix=self.task_prefix, max_keys=limit)
            task_ids = []
            
            for obj in objects:
                # Extract task ID from key (remove prefix and .json extension)
                task_id = obj.key.replace(self.task_prefix, '').replace('.json', '')
                task_ids.append(task_id)
            
            logger.info("Listed stored tasks from Spinal Cord", count=len(task_ids))
            return task_ids
            
        except Exception as e:
            logger.error("Error listing stored tasks from Spinal Cord", error=str(e))
            raise
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old tasks from fallback storage.
        
        Args:
            max_age_hours: Maximum age of tasks to keep (in hours)
            
        Returns:
            Number of tasks cleaned up
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            objects = await self.r2_client.list_objects(prefix=self.task_prefix)
            
            cleaned_count = 0
            for obj in objects:
                if obj.last_modified < cutoff_time:
                    success = await self.r2_client.delete_object(obj.key)
                    if success:
                        cleaned_count += 1
            
            logger.info("Cleaned up old tasks from Spinal Cord", count=cleaned_count, max_age_hours=max_age_hours)
            return cleaned_count
            
        except Exception as e:
            logger.error("Error cleaning up old tasks from Spinal Cord", error=str(e))
            raise


# Global instance for dependency injection
_spinal_cord_storage: Optional[SpinalCordStorage] = None


async def get_spinal_cord_storage() -> SpinalCordStorage:
    """
    Get SpinalCordStorage instance for dependency injection.
    
    Usage in FastAPI:
        @app.post("/fallback/store")
        async def store_task(
            task_data: dict,
            storage: SpinalCordStorage = Depends(get_spinal_cord_storage)
        ):
            # Use storage here
            pass
    """
    global _spinal_cord_storage
    if _spinal_cord_storage is None:
        _spinal_cord_storage = SpinalCordStorage()
    return _spinal_cord_storage