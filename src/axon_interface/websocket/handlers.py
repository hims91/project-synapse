"""
WebSocket message handlers.

Handles incoming WebSocket messages and routes them to appropriate handlers
based on message type and user permissions.
"""

import logging
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from .events import WebSocketEvent, EventType, create_user_notification_event
from .auth import WebSocketAuthenticator
from .channels import ChannelManager


logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket message processing and routing."""
    
    def __init__(self, authenticator: WebSocketAuthenticator, channel_manager: ChannelManager):
        self.authenticator = authenticator
        self.channel_manager = channel_manager
        self.message_handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.register_handler('ping', self._handle_ping)
        self.register_handler('subscribe', self._handle_subscribe)
        self.register_handler('unsubscribe', self._handle_unsubscribe)
        self.register_handler('get_channels', self._handle_get_channels)
        self.register_handler('get_history', self._handle_get_history)
        self.register_handler('authenticate', self._handle_authenticate)
        self.register_handler('get_status', self._handle_get_status)
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a message handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    def register_middleware(self, middleware: Callable):
        """
        Register middleware to process messages before handlers.
        
        Args:
            middleware: Middleware function
        """
        self.middleware.append(middleware)
        logger.debug("Registered middleware function")
    
    async def handle_message(
        self, 
        connection_id: str, 
        message: str
    ) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming WebSocket message.
        
        Args:
            connection_id: Connection identifier
            message: Raw message string
            
        Returns:
            Response message or None
        """
        try:
            # Parse message
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from connection {connection_id}: {e}")
                return self._create_error_response("Invalid JSON format")
            
            # Validate message structure
            if not isinstance(data, dict) or 'type' not in data:
                logger.warning(f"Invalid message structure from connection {connection_id}")
                return self._create_error_response("Invalid message structure")
            
            message_type = data['type']
            message_id = data.get('id', f"msg_{int(datetime.utcnow().timestamp())}")
            payload = data.get('data', {})
            
            # Update connection activity
            self.authenticator.update_activity(connection_id)
            
            # Apply middleware
            for middleware_func in self.middleware:
                try:
                    result = await middleware_func(connection_id, message_type, payload)
                    if result is False:  # Middleware rejected the message
                        return self._create_error_response("Message rejected by middleware")
                except Exception as e:
                    logger.error(f"Middleware error: {e}")
                    return self._create_error_response("Internal server error")
            
            # Route to handler
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                try:
                    response = await handler(connection_id, payload, message_id)
                    return response
                except Exception as e:
                    logger.error(f"Handler error for {message_type}: {e}")
                    return self._create_error_response(
                        "Handler error", 
                        message_id=message_id
                    )
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return self._create_error_response(
                    f"Unknown message type: {message_type}",
                    message_id=message_id
                )
        
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            return self._create_error_response("Internal server error")
    
    async def _handle_ping(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle ping message."""
        return {
            'type': 'pong',
            'id': message_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {'message': 'pong'}
        }
    
    async def _handle_authenticate(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle authentication message."""
        token = payload.get('token')
        api_key = payload.get('api_key')
        
        user_info = await self.authenticator.authenticate_connection(
            connection_id, token, api_key
        )
        
        if user_info:
            return {
                'type': 'auth_success',
                'id': message_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'user_id': user_info['user_id'],
                    'tier': user_info.get('tier', 'free'),
                    'permissions': user_info.get('permissions', [])
                }
            }
        else:
            return self._create_error_response(
                "Authentication failed",
                message_id=message_id
            )
    
    async def _handle_subscribe(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle channel subscription."""
        if not self.authenticator.is_authenticated(connection_id):
            return self._create_error_response(
                "Authentication required",
                message_id=message_id
            )
        
        channel = payload.get('channel')
        if not channel:
            return self._create_error_response(
                "Channel name required",
                message_id=message_id
            )
        
        # Check channel access permissions
        if not self.authenticator.subscribe_to_channel(connection_id, channel):
            return self._create_error_response(
                f"Access denied to channel: {channel}",
                message_id=message_id
            )
        
        # Subscribe to channel
        success = self.channel_manager.subscribe_connection(connection_id, channel)
        
        if success:
            # Send recent history if requested
            include_history = payload.get('include_history', False)
            history = []
            if include_history:
                history = self.channel_manager.get_channel_history(channel, limit=10)
            
            return {
                'type': 'subscribed',
                'id': message_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'channel': channel,
                    'history': history
                }
            }
        else:
            return self._create_error_response(
                f"Failed to subscribe to channel: {channel}",
                message_id=message_id
            )
    
    async def _handle_unsubscribe(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle channel unsubscription."""
        channel = payload.get('channel')
        if not channel:
            return self._create_error_response(
                "Channel name required",
                message_id=message_id
            )
        
        # Unsubscribe from channel
        self.authenticator.unsubscribe_from_channel(connection_id, channel)
        self.channel_manager.unsubscribe_connection(connection_id, channel)
        
        return {
            'type': 'unsubscribed',
            'id': message_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {'channel': channel}
        }
    
    async def _handle_get_channels(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle get channels request."""
        if not self.authenticator.is_authenticated(connection_id):
            return self._create_error_response(
                "Authentication required",
                message_id=message_id
            )
        
        # Get subscribed channels
        subscribed_channels = self.authenticator.get_subscribed_channels(connection_id)
        
        # Get available channels (filtered by permissions)
        all_channels = self.channel_manager.list_channels()
        available_channels = []
        
        for channel_info in all_channels:
            channel_name = channel_info['name']
            if self.authenticator._can_access_channel(connection_id, channel_name):
                available_channels.append(channel_info)
        
        return {
            'type': 'channels',
            'id': message_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'subscribed': list(subscribed_channels),
                'available': available_channels
            }
        }
    
    async def _handle_get_history(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle get channel history request."""
        if not self.authenticator.is_authenticated(connection_id):
            return self._create_error_response(
                "Authentication required",
                message_id=message_id
            )
        
        channel = payload.get('channel')
        limit = payload.get('limit', 50)
        
        if not channel:
            return self._create_error_response(
                "Channel name required",
                message_id=message_id
            )
        
        # Check if user has access to channel
        if not self.authenticator._can_access_channel(connection_id, channel):
            return self._create_error_response(
                f"Access denied to channel: {channel}",
                message_id=message_id
            )
        
        history = self.channel_manager.get_channel_history(channel, limit)
        
        return {
            'type': 'history',
            'id': message_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'channel': channel,
                'messages': history
            }
        }
    
    async def _handle_get_status(
        self, 
        connection_id: str, 
        payload: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Handle get connection status request."""
        user_info = self.authenticator.get_user_info(connection_id)
        subscribed_channels = self.authenticator.get_subscribed_channels(connection_id)
        
        return {
            'type': 'status',
            'id': message_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'connection_id': connection_id,
                'authenticated': user_info is not None,
                'user_id': user_info.get('user_id') if user_info else None,
                'user_tier': user_info.get('user_tier') if user_info else None,
                'subscribed_channels': list(subscribed_channels),
                'last_activity': user_info.get('last_activity').isoformat() if user_info else None
            }
        }
    
    def _create_error_response(
        self, 
        error_message: str, 
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            'type': 'error',
            'id': message_id,
            'timestamp': datetime.utcnow().isoformat(),
            'error': {
                'message': error_message,
                'code': 'WEBSOCKET_ERROR'
            }
        }
    
    async def broadcast_event(
        self, 
        event: WebSocketEvent, 
        target_connections: Optional[set] = None
    ) -> int:
        """
        Broadcast an event to connections.
        
        Args:
            event: Event to broadcast
            target_connections: Specific connections to target, or None for channel-based routing
            
        Returns:
            Number of connections that received the event
        """
        if target_connections:
            # Direct connection targeting
            sent_count = 0
            for connection_id in target_connections:
                if self.channel_manager.send_to_connection(connection_id, event):
                    sent_count += 1
            return sent_count
        
        elif event.channel:
            # Channel-based broadcasting
            connections = self.channel_manager.broadcast_to_channel(event.channel, event)
            return len(connections)
        
        elif event.user_id:
            # User-specific event
            user_connections = self.authenticator.get_user_connections(event.user_id)
            sent_count = 0
            for connection_id in user_connections:
                if self.channel_manager.send_to_connection(connection_id, event):
                    sent_count += 1
            return sent_count
        
        else:
            logger.warning(f"Event {event.event_id} has no routing information")
            return 0
    
    async def send_notification(
        self, 
        user_id: str, 
        title: str, 
        message: str, 
        notification_type: str = "info",
        priority: str = "normal",
        action_url: Optional[str] = None
    ) -> bool:
        """
        Send a notification to a specific user.
        
        Args:
            user_id: Target user ID
            title: Notification title
            message: Notification message
            notification_type: Type of notification
            priority: Notification priority
            action_url: Optional action URL
            
        Returns:
            True if notification was sent, False otherwise
        """
        notification_event = create_user_notification_event(
            notification_id=f"notif_{int(datetime.utcnow().timestamp())}",
            title=title,
            message=message,
            notification_type=notification_type,
            user_id=user_id,
            priority=priority,
            action_url=action_url
        )
        
        sent_count = await self.broadcast_event(notification_event)
        return sent_count > 0
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            'registered_handlers': list(self.message_handlers.keys()),
            'middleware_count': len(self.middleware),
            'connection_stats': self.authenticator.get_connection_stats(),
            'channel_stats': self.channel_manager.get_statistics()
        }