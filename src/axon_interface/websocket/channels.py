"""
WebSocket channel management.

Manages WebSocket channels for organizing and routing messages to specific
groups of connected clients.
"""

import logging
from typing import Dict, Set, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from .events import WebSocketEvent, EventType


logger = logging.getLogger(__name__)


class ChannelManager:
    """Manages WebSocket channels and message routing."""
    
    def __init__(self):
        self.channels: Dict[str, Set[str]] = defaultdict(set)  # channel -> connection_ids
        self.connection_channels: Dict[str, Set[str]] = defaultdict(set)  # connection_id -> channels
        self.channel_metadata: Dict[str, Dict[str, Any]] = {}
        self.message_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history_per_channel = 100
    
    def create_channel(
        self, 
        channel_name: str, 
        description: Optional[str] = None,
        persistent: bool = False,
        max_connections: Optional[int] = None
    ) -> bool:
        """
        Create a new channel.
        
        Args:
            channel_name: Name of the channel
            description: Optional channel description
            persistent: Whether the channel persists when empty
            max_connections: Maximum number of connections allowed
            
        Returns:
            True if channel was created, False if it already exists
        """
        if channel_name in self.channel_metadata:
            logger.warning(f"Channel {channel_name} already exists")
            return False
        
        self.channel_metadata[channel_name] = {
            'created_at': datetime.utcnow(),
            'description': description,
            'persistent': persistent,
            'max_connections': max_connections,
            'message_count': 0,
            'last_activity': datetime.utcnow()
        }
        
        # Initialize empty channel
        if channel_name not in self.channels:
            self.channels[channel_name] = set()
        
        logger.info(f"Created channel: {channel_name}")
        return True
    
    def delete_channel(self, channel_name: str) -> bool:
        """
        Delete a channel and disconnect all subscribers.
        
        Args:
            channel_name: Name of the channel to delete
            
        Returns:
            True if channel was deleted, False if it didn't exist
        """
        if channel_name not in self.channels:
            return False
        
        # Disconnect all subscribers
        connections = self.channels[channel_name].copy()
        for connection_id in connections:
            self.unsubscribe_connection(connection_id, channel_name)
        
        # Remove channel
        del self.channels[channel_name]
        if channel_name in self.channel_metadata:
            del self.channel_metadata[channel_name]
        if channel_name in self.message_history:
            del self.message_history[channel_name]
        
        logger.info(f"Deleted channel: {channel_name}")
        return True
    
    def subscribe_connection(self, connection_id: str, channel_name: str) -> bool:
        """
        Subscribe a connection to a channel.
        
        Args:
            connection_id: Connection identifier
            channel_name: Channel to subscribe to
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Check if channel exists or create it
        if channel_name not in self.channels:
            self.create_channel(channel_name, persistent=False)
        
        # Check connection limit
        channel_meta = self.channel_metadata.get(channel_name, {})
        max_connections = channel_meta.get('max_connections')
        if max_connections and len(self.channels[channel_name]) >= max_connections:
            logger.warning(f"Channel {channel_name} is at capacity ({max_connections})")
            return False
        
        # Add connection to channel
        self.channels[channel_name].add(connection_id)
        self.connection_channels[connection_id].add(channel_name)
        
        # Update channel metadata
        if channel_name in self.channel_metadata:
            self.channel_metadata[channel_name]['last_activity'] = datetime.utcnow()
        
        logger.debug(f"Connection {connection_id} subscribed to channel {channel_name}")
        return True
    
    def unsubscribe_connection(self, connection_id: str, channel_name: str) -> bool:
        """
        Unsubscribe a connection from a channel.
        
        Args:
            connection_id: Connection identifier
            channel_name: Channel to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if channel_name not in self.channels:
            return False
        
        # Remove connection from channel
        self.channels[channel_name].discard(connection_id)
        self.connection_channels[connection_id].discard(channel_name)
        
        # Clean up empty non-persistent channels
        channel_meta = self.channel_metadata.get(channel_name, {})
        if (not self.channels[channel_name] and 
            not channel_meta.get('persistent', False)):
            self.delete_channel(channel_name)
        
        logger.debug(f"Connection {connection_id} unsubscribed from channel {channel_name}")
        return True
    
    def disconnect_connection(self, connection_id: str):
        """
        Disconnect a connection from all channels.
        
        Args:
            connection_id: Connection identifier
        """
        channels = self.connection_channels[connection_id].copy()
        for channel_name in channels:
            self.unsubscribe_connection(connection_id, channel_name)
        
        # Clean up connection tracking
        if connection_id in self.connection_channels:
            del self.connection_channels[connection_id]
        
        logger.debug(f"Connection {connection_id} disconnected from all channels")
    
    def get_channel_connections(self, channel_name: str) -> Set[str]:
        """
        Get all connections subscribed to a channel.
        
        Args:
            channel_name: Channel name
            
        Returns:
            Set of connection IDs
        """
        return self.channels.get(channel_name, set()).copy()
    
    def get_connection_channels(self, connection_id: str) -> Set[str]:
        """
        Get all channels a connection is subscribed to.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Set of channel names
        """
        return self.connection_channels.get(connection_id, set()).copy()
    
    def broadcast_to_channel(
        self, 
        channel_name: str, 
        event: WebSocketEvent,
        exclude_connections: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Broadcast an event to all connections in a channel.
        
        Args:
            channel_name: Channel to broadcast to
            event: Event to broadcast
            exclude_connections: Connections to exclude from broadcast
            
        Returns:
            Set of connection IDs that received the message
        """
        if channel_name not in self.channels:
            logger.warning(f"Attempted to broadcast to non-existent channel: {channel_name}")
            return set()
        
        connections = self.channels[channel_name].copy()
        if exclude_connections:
            connections -= exclude_connections
        
        # Store message in history
        self._store_message_history(channel_name, event)
        
        # Update channel metadata
        if channel_name in self.channel_metadata:
            self.channel_metadata[channel_name]['message_count'] += 1
            self.channel_metadata[channel_name]['last_activity'] = datetime.utcnow()
        
        logger.debug(f"Broadcasting event {event.event_type} to {len(connections)} connections in channel {channel_name}")
        return connections
    
    def send_to_connection(
        self, 
        connection_id: str, 
        event: WebSocketEvent,
        channel_name: Optional[str] = None
    ) -> bool:
        """
        Send an event to a specific connection.
        
        Args:
            connection_id: Target connection
            event: Event to send
            channel_name: Optional channel context
            
        Returns:
            True if connection exists, False otherwise
        """
        # Check if connection exists in any channel
        if connection_id not in self.connection_channels:
            logger.warning(f"Attempted to send to non-existent connection: {connection_id}")
            return False
        
        # If channel specified, store in history
        if channel_name:
            self._store_message_history(channel_name, event)
        
        logger.debug(f"Sending event {event.event_type} to connection {connection_id}")
        return True
    
    def _store_message_history(self, channel_name: str, event: WebSocketEvent):
        """Store message in channel history."""
        history = self.message_history[channel_name]
        
        # Add message to history
        history.append({
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'user_id': event.user_id,
            'data': event.data
        })
        
        # Trim history if too long
        if len(history) > self.max_history_per_channel:
            history[:] = history[-self.max_history_per_channel:]
    
    def get_channel_history(
        self, 
        channel_name: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get message history for a channel.
        
        Args:
            channel_name: Channel name
            limit: Maximum number of messages to return
            
        Returns:
            List of historical messages
        """
        history = self.message_history.get(channel_name, [])
        
        if limit:
            return history[-limit:]
        return history.copy()
    
    def get_channel_info(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a channel.
        
        Args:
            channel_name: Channel name
            
        Returns:
            Channel information or None if channel doesn't exist
        """
        if channel_name not in self.channels:
            return None
        
        metadata = self.channel_metadata.get(channel_name, {})
        return {
            'name': channel_name,
            'connection_count': len(self.channels[channel_name]),
            'created_at': metadata.get('created_at'),
            'description': metadata.get('description'),
            'persistent': metadata.get('persistent', False),
            'max_connections': metadata.get('max_connections'),
            'message_count': metadata.get('message_count', 0),
            'last_activity': metadata.get('last_activity')
        }
    
    def list_channels(self) -> List[Dict[str, Any]]:
        """
        List all channels with their information.
        
        Returns:
            List of channel information dictionaries
        """
        channels = []
        for channel_name in self.channels:
            info = self.get_channel_info(channel_name)
            if info:
                channels.append(info)
        
        return sorted(channels, key=lambda x: x['name'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get channel manager statistics.
        
        Returns:
            Statistics dictionary
        """
        total_channels = len(self.channels)
        total_connections = len(self.connection_channels)
        total_subscriptions = sum(len(channels) for channels in self.connection_channels.values())
        
        # Channel size distribution
        channel_sizes = [len(connections) for connections in self.channels.values()]
        avg_channel_size = sum(channel_sizes) / len(channel_sizes) if channel_sizes else 0
        
        # Connection subscription distribution
        subscription_counts = [len(channels) for channels in self.connection_channels.values()]
        avg_subscriptions_per_connection = (
            sum(subscription_counts) / len(subscription_counts) 
            if subscription_counts else 0
        )
        
        return {
            'total_channels': total_channels,
            'total_connections': total_connections,
            'total_subscriptions': total_subscriptions,
            'average_channel_size': avg_channel_size,
            'average_subscriptions_per_connection': avg_subscriptions_per_connection,
            'largest_channel_size': max(channel_sizes) if channel_sizes else 0,
            'total_messages_stored': sum(
                len(history) for history in self.message_history.values()
            )
        }
    
    def cleanup_empty_channels(self) -> int:
        """
        Clean up empty non-persistent channels.
        
        Returns:
            Number of channels cleaned up
        """
        empty_channels = []
        
        for channel_name, connections in self.channels.items():
            if not connections:
                metadata = self.channel_metadata.get(channel_name, {})
                if not metadata.get('persistent', False):
                    empty_channels.append(channel_name)
        
        for channel_name in empty_channels:
            self.delete_channel(channel_name)
        
        logger.info(f"Cleaned up {len(empty_channels)} empty channels")
        return len(empty_channels)