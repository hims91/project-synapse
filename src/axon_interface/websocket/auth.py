"""
WebSocket authentication and authorization.

Handles authentication of WebSocket connections and manages user-specific
channels and permissions.
"""

import logging
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, status

try:
    from ...shared.config import get_settings
except ImportError:
    # Fallback for testing
    def get_settings():
        class MockSettings:
            class Security:
                secret_key = "test-secret-key"
            security = Security()
        return MockSettings()


logger = logging.getLogger(__name__)


class WebSocketAuthenticator:
    """Handles WebSocket connection authentication and authorization."""
    
    def __init__(self):
        self.settings = get_settings()
        self.authenticated_connections: Dict[str, Dict[str, Any]] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        
    async def authenticate_connection(
        self, 
        connection_id: str, 
        token: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate a WebSocket connection using JWT token or API key.
        
        Args:
            connection_id: Unique connection identifier
            token: JWT authentication token
            api_key: API key for authentication
            
        Returns:
            User information if authentication successful, None otherwise
        """
        try:
            user_info = None
            
            if token:
                user_info = await self._authenticate_jwt_token(token)
            elif api_key:
                user_info = await self._authenticate_api_key(api_key)
            else:
                logger.warning(f"No authentication provided for connection {connection_id}")
                return None
            
            if user_info:
                # Store authenticated connection
                self.authenticated_connections[connection_id] = {
                    'user_id': user_info['user_id'],
                    'user_tier': user_info.get('tier', 'free'),
                    'authenticated_at': datetime.utcnow(),
                    'permissions': user_info.get('permissions', []),
                    'channels': set(),
                    'last_activity': datetime.utcnow()
                }
                
                # Track user connections
                user_id = user_info['user_id']
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
                
                logger.info(f"WebSocket connection {connection_id} authenticated for user {user_id}")
                return user_info
            
        except Exception as e:
            logger.error(f"Authentication error for connection {connection_id}: {e}")
            
        return None
    
    async def _authenticate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate using JWT token."""
        try:
            # Decode JWT token
            payload = jwt.decode(
                token, 
                self.settings.security.secret_key, 
                algorithms=["HS256"]
            )
            
            # Check token expiration
            exp = payload.get('exp')
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                logger.warning("JWT token expired")
                return None
            
            return {
                'user_id': payload.get('sub'),
                'tier': payload.get('tier', 'free'),
                'permissions': payload.get('permissions', []),
                'auth_method': 'jwt'
            }
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def _authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate using API key."""
        # Mock API key validation - in production, this would check against database
        valid_api_keys = {
            'test-api-key-123': {
                'user_id': 'user_123',
                'tier': 'premium',
                'permissions': ['read', 'write', 'websocket']
            },
            'demo-api-key-456': {
                'user_id': 'demo_user',
                'tier': 'free',
                'permissions': ['read', 'websocket']
            }
        }
        
        if api_key in valid_api_keys:
            user_info = valid_api_keys[api_key].copy()
            user_info['auth_method'] = 'api_key'
            return user_info
        
        logger.warning(f"Invalid API key: {api_key}")
        return None
    
    def is_authenticated(self, connection_id: str) -> bool:
        """Check if a connection is authenticated."""
        return connection_id in self.authenticated_connections
    
    def get_user_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get user information for an authenticated connection."""
        return self.authenticated_connections.get(connection_id)
    
    def get_user_id(self, connection_id: str) -> Optional[str]:
        """Get user ID for a connection."""
        conn_info = self.authenticated_connections.get(connection_id)
        return conn_info['user_id'] if conn_info else None
    
    def get_user_tier(self, connection_id: str) -> str:
        """Get user tier for a connection."""
        conn_info = self.authenticated_connections.get(connection_id)
        return conn_info.get('user_tier', 'free') if conn_info else 'free'
    
    def has_permission(self, connection_id: str, permission: str) -> bool:
        """Check if a connection has a specific permission."""
        conn_info = self.authenticated_connections.get(connection_id)
        if not conn_info:
            return False
        
        permissions = conn_info.get('permissions', [])
        return permission in permissions or 'admin' in permissions
    
    def update_activity(self, connection_id: str):
        """Update last activity timestamp for a connection."""
        if connection_id in self.authenticated_connections:
            self.authenticated_connections[connection_id]['last_activity'] = datetime.utcnow()
    
    def subscribe_to_channel(self, connection_id: str, channel: str) -> bool:
        """Subscribe a connection to a channel."""
        if connection_id not in self.authenticated_connections:
            return False
        
        # Check channel permissions
        if not self._can_access_channel(connection_id, channel):
            logger.warning(f"Connection {connection_id} denied access to channel {channel}")
            return False
        
        self.authenticated_connections[connection_id]['channels'].add(channel)
        logger.info(f"Connection {connection_id} subscribed to channel {channel}")
        return True
    
    def unsubscribe_from_channel(self, connection_id: str, channel: str):
        """Unsubscribe a connection from a channel."""
        if connection_id in self.authenticated_connections:
            self.authenticated_connections[connection_id]['channels'].discard(channel)
            logger.info(f"Connection {connection_id} unsubscribed from channel {channel}")
    
    def get_subscribed_channels(self, connection_id: str) -> Set[str]:
        """Get channels a connection is subscribed to."""
        conn_info = self.authenticated_connections.get(connection_id)
        return conn_info['channels'] if conn_info else set()
    
    def _can_access_channel(self, connection_id: str, channel: str) -> bool:
        """Check if a connection can access a specific channel."""
        conn_info = self.authenticated_connections.get(connection_id)
        if not conn_info:
            return False
        
        user_id = conn_info['user_id']
        user_tier = conn_info['user_tier']
        permissions = conn_info.get('permissions', [])
        
        # Channel access rules
        if channel.startswith('user:'):
            # User-specific channels
            channel_user_id = channel.split(':', 1)[1]
            return user_id == channel_user_id or 'admin' in permissions
        
        elif channel.startswith('tier:'):
            # Tier-specific channels
            required_tier = channel.split(':', 1)[1]
            tier_hierarchy = {'free': 0, 'premium': 1, 'enterprise': 2}
            user_tier_level = tier_hierarchy.get(user_tier, 0)
            required_tier_level = tier_hierarchy.get(required_tier, 0)
            return user_tier_level >= required_tier_level
        
        elif channel in ['system', 'alerts', 'monitoring']:
            # System channels require specific permissions
            return 'admin' in permissions or 'monitoring' in permissions
        
        elif channel in ['jobs', 'feeds', 'analytics']:
            # General channels available to authenticated users
            return True
        
        else:
            # Unknown channels denied by default
            return False
    
    def disconnect_user(self, connection_id: str):
        """Handle user disconnection."""
        if connection_id in self.authenticated_connections:
            user_id = self.authenticated_connections[connection_id]['user_id']
            
            # Remove from authenticated connections
            del self.authenticated_connections[connection_id]
            
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            logger.info(f"WebSocket connection {connection_id} disconnected for user {user_id}")
    
    def get_user_connections(self, user_id: str) -> Set[str]:
        """Get all connection IDs for a user."""
        return self.user_connections.get(user_id, set()).copy()
    
    def get_connections_for_channel(self, channel: str) -> Set[str]:
        """Get all connection IDs subscribed to a channel."""
        connections = set()
        
        for connection_id, conn_info in self.authenticated_connections.items():
            if channel in conn_info['channels']:
                connections.add(connection_id)
        
        return connections
    
    def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """Clean up inactive connections."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        inactive_connections = []
        
        for connection_id, conn_info in self.authenticated_connections.items():
            if conn_info['last_activity'] < cutoff_time:
                inactive_connections.append(connection_id)
        
        for connection_id in inactive_connections:
            logger.info(f"Cleaning up inactive connection {connection_id}")
            self.disconnect_user(connection_id)
        
        return len(inactive_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        total_connections = len(self.authenticated_connections)
        unique_users = len(self.user_connections)
        
        # Count by tier
        tier_counts = {}
        channel_counts = {}
        
        for conn_info in self.authenticated_connections.values():
            tier = conn_info['user_tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            for channel in conn_info['channels']:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        return {
            'total_connections': total_connections,
            'unique_users': unique_users,
            'tier_distribution': tier_counts,
            'channel_subscriptions': channel_counts,
            'average_channels_per_connection': (
                sum(len(info['channels']) for info in self.authenticated_connections.values()) 
                / total_connections if total_connections > 0 else 0
            )
        }