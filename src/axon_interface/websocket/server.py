"""
WebSocket server implementation.

Main WebSocket server that manages connections, authentication, and message routing
for real-time communication with clients.
"""

import logging
import asyncio
import json
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timedelta
from uuid import uuid4
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from .auth import WebSocketAuthenticator
from .channels import ChannelManager
from .handlers import WebSocketHandler
from .events import (
    WebSocketEvent, EventType, 
    create_job_event, create_system_health_event, create_alert_event
)


logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.last_pong = datetime.utcnow()
        self.message_count = 0
        self.is_alive = True
        self.client_info = {}
    
    async def send_json(self, data: Dict[str, Any]) -> bool:
        """Send JSON data to the WebSocket."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(data)
                return True
        except Exception as e:
            logger.error(f"Error sending to connection {self.connection_id}: {e}")
            self.is_alive = False
        return False
    
    async def send_text(self, text: str) -> bool:
        """Send text data to the WebSocket."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_text(text)
                return True
        except Exception as e:
            logger.error(f"Error sending to connection {self.connection_id}: {e}")
            self.is_alive = False
        return False
    
    async def close(self, code: int = 1000, reason: str = ""):
        """Close the WebSocket connection."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.error(f"Error closing connection {self.connection_id}: {e}")
        finally:
            self.is_alive = False


class WebSocketManager:
    """Manages WebSocket connections and real-time communication."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.authenticator = WebSocketAuthenticator()
        self.channel_manager = ChannelManager()
        self.handler = WebSocketHandler(self.authenticator, self.channel_manager)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.start_time = datetime.utcnow()
        
        # Initialize default channels
        self._initialize_default_channels()
        
        # Background tasks will be started when first connection is made
        self._background_tasks_started = False
    
    def _initialize_default_channels(self):
        """Initialize default system channels."""
        default_channels = [
            ('system', 'System-wide notifications', True),
            ('jobs', 'Job status updates', True),
            ('alerts', 'System alerts and monitoring', True),
            ('feeds', 'Feed update notifications', True),
            ('analytics', 'Analytics and metrics updates', True),
        ]
        
        for channel_name, description, persistent in default_channels:
            self.channel_manager.create_channel(
                channel_name, 
                description=description, 
                persistent=persistent
            )
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        tasks = [
            self._heartbeat_task(),
            self._cleanup_task(),
            self._health_monitoring_task()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            
        Returns:
            Connection ID
        """
        # Start background tasks on first connection
        if not self._background_tasks_started:
            self._start_background_tasks()
            self._background_tasks_started = True
        
        await websocket.accept()
        
        connection_id = str(uuid4())
        connection = WebSocketConnection(websocket, connection_id)
        
        # Extract client info
        connection.client_info = {
            'client_host': websocket.client.host if websocket.client else 'unknown',
            'client_port': websocket.client.port if websocket.client else 0,
            'user_agent': websocket.headers.get('user-agent', 'unknown'),
            'origin': websocket.headers.get('origin', 'unknown')
        }
        
        self.connections[connection_id] = connection
        self.total_connections += 1
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send connection established event
        await self._send_connection_event(connection_id, EventType.CONNECTION_ESTABLISHED)
        
        return connection_id
    
    async def disconnect(self, connection_id: str, code: int = 1000, reason: str = ""):
        """
        Handle WebSocket disconnection.
        
        Args:
            connection_id: Connection identifier
            code: Close code
            reason: Close reason
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Clean up authentication and channels
        self.authenticator.disconnect_user(connection_id)
        self.channel_manager.disconnect_connection(connection_id)
        
        # Close connection
        await connection.close(code, reason)
        
        # Remove from connections
        del self.connections[connection_id]
        
        logger.info(f"WebSocket connection closed: {connection_id} (code: {code}, reason: {reason})")
    
    async def handle_message(self, connection_id: str, message: str):
        """
        Handle incoming WebSocket message.
        
        Args:
            connection_id: Connection identifier
            message: Raw message string
        """
        if connection_id not in self.connections:
            logger.warning(f"Message from unknown connection: {connection_id}")
            return
        
        connection = self.connections[connection_id]
        connection.message_count += 1
        self.total_messages += 1
        
        # Process message through handler
        response = await self.handler.handle_message(connection_id, message)
        
        # Send response if provided
        if response:
            await connection.send_json(response)
    
    async def broadcast_event(
        self, 
        event: WebSocketEvent, 
        target_connections: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast an event to connections.
        
        Args:
            event: Event to broadcast
            target_connections: Specific connections to target
            
        Returns:
            Number of connections that received the event
        """
        if target_connections:
            # Direct connection targeting
            sent_count = 0
            for connection_id in target_connections:
                if await self._send_to_connection(connection_id, event):
                    sent_count += 1
            return sent_count
        
        # Use handler for routing
        return await self.handler.broadcast_event(event, target_connections)
    
    async def _send_to_connection(self, connection_id: str, event: WebSocketEvent) -> bool:
        """Send an event to a specific connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        event_data = {
            'type': 'event',
            'event_type': event.event_type,
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'data': event.data,
            'metadata': event.metadata
        }
        
        return await connection.send_json(event_data)
    
    async def _send_connection_event(self, connection_id: str, event_type: EventType):
        """Send a connection-related event."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        event_data = {
            'type': 'connection',
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'connection_id': connection_id,
                'client_info': connection.client_info
            }
        }
        
        await connection.send_json(event_data)
    
    async def send_job_update(
        self, 
        job_id: str, 
        job_type: str, 
        status: str,
        user_id: Optional[str] = None,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ):
        """Send a job status update."""
        event_type_map = {
            'pending': EventType.JOB_CREATED,
            'processing': EventType.JOB_STARTED,
            'completed': EventType.JOB_COMPLETED,
            'failed': EventType.JOB_FAILED,
            'cancelled': EventType.JOB_CANCELLED
        }
        
        event_type = event_type_map.get(status, EventType.JOB_PROGRESS)
        
        job_event = create_job_event(
            event_type=event_type,
            job_id=job_id,
            job_type=job_type,
            status=status,
            user_id=user_id,
            progress=progress,
            error_message=error_message,
            result_data=result_data
        )
        
        # Set channel for job events
        job_event.channel = 'jobs'
        
        await self.broadcast_event(job_event)
        logger.debug(f"Sent job update: {job_id} -> {status}")
    
    async def send_system_health_update(
        self, 
        component: str, 
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        issues: Optional[List[str]] = None
    ):
        """Send a system health update."""
        health_event = create_system_health_event(
            component=component,
            status=status,
            metrics=metrics,
            issues=issues
        )
        
        health_event.channel = 'system'
        
        await self.broadcast_event(health_event)
        logger.debug(f"Sent health update: {component} -> {status}")
    
    async def send_alert(
        self, 
        alert_id: str, 
        severity: str, 
        title: str, 
        message: str,
        source: str,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ):
        """Send an alert notification."""
        alert_event = create_alert_event(
            event_type=EventType.ALERT_CREATED,
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            tags=tags,
            user_id=user_id
        )
        
        alert_event.channel = 'alerts'
        
        await self.broadcast_event(alert_event)
        logger.info(f"Sent alert: {alert_id} ({severity}) - {title}")
    
    async def _heartbeat_task(self):
        """Background task for connection heartbeat."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                dead_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Send ping
                    ping_data = {
                        'type': 'ping',
                        'timestamp': current_time.isoformat()
                    }
                    
                    if not await connection.send_json(ping_data):
                        dead_connections.append(connection_id)
                    else:
                        connection.last_ping = current_time
                
                # Clean up dead connections
                for connection_id in dead_connections:
                    await self.disconnect(connection_id, code=1001, reason="Connection lost")
                
                # Wait before next heartbeat
                await asyncio.sleep(30)  # 30 second heartbeat
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_task(self):
        """Background task for periodic cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up inactive connections
                inactive_count = self.authenticator.cleanup_inactive_connections(timeout_minutes=60)
                if inactive_count > 0:
                    logger.info(f"Cleaned up {inactive_count} inactive connections")
                
                # Clean up empty channels
                empty_channels = self.channel_manager.cleanup_empty_channels()
                if empty_channels > 0:
                    logger.info(f"Cleaned up {empty_channels} empty channels")
                
                # Wait before next cleanup
                await asyncio.sleep(300)  # 5 minute cleanup interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitoring_task(self):
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Send periodic health updates
                await self.send_system_health_update(
                    component='websocket_server',
                    status='healthy',
                    metrics={
                        'active_connections': len(self.connections),
                        'total_connections': self.total_connections,
                        'total_messages': self.total_messages,
                        'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
                    }
                )
                
                # Wait before next health update
                await asyncio.sleep(60)  # 1 minute health updates
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Shutdown the WebSocket manager."""
        logger.info("Shutting down WebSocket manager...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close all connections
        close_tasks = []
        for connection_id in list(self.connections.keys()):
            close_tasks.append(self.disconnect(connection_id, code=1001, reason="Server shutdown"))
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        logger.info("WebSocket manager shutdown complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WebSocket server statistics."""
        uptime = datetime.utcnow() - self.start_time
        
        return {
            'server': {
                'uptime_seconds': uptime.total_seconds(),
                'start_time': self.start_time.isoformat(),
                'active_connections': len(self.connections),
                'total_connections': self.total_connections,
                'total_messages': self.total_messages,
                'background_tasks': len(self._background_tasks)
            },
            'authentication': self.authenticator.get_connection_stats(),
            'channels': self.channel_manager.get_statistics(),
            'handlers': self.handler.get_handler_stats()
        }
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        user_info = self.authenticator.get_user_info(connection_id)
        channels = self.authenticator.get_subscribed_channels(connection_id)
        
        return {
            'connection_id': connection_id,
            'connected_at': connection.connected_at.isoformat(),
            'last_ping': connection.last_ping.isoformat(),
            'message_count': connection.message_count,
            'is_alive': connection.is_alive,
            'client_info': connection.client_info,
            'user_info': user_info,
            'subscribed_channels': list(channels)
        }


# Global WebSocket manager instance
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager