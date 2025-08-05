"""
WebSocket module for real-time communication.

This module provides WebSocket server functionality for real-time updates
including job status notifications, monitoring alerts, and system health updates.
"""

from .server import WebSocketManager, get_websocket_manager
from .handlers import WebSocketHandler
from .auth import WebSocketAuthenticator
from .channels import ChannelManager
from .events import EventType, WebSocketEvent

__all__ = [
    'WebSocketManager',
    'get_websocket_manager',
    'WebSocketHandler',
    'WebSocketAuthenticator',
    'ChannelManager',
    'EventType',
    'WebSocketEvent',
]