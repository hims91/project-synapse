"""
WebSocket API endpoints.

Provides WebSocket endpoints for real-time communication with clients.
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional

from ..websocket import get_websocket_manager, WebSocketManager


logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT authentication token"),
    api_key: Optional[str] = Query(None, description="API key for authentication"),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Main WebSocket endpoint for real-time communication.
    
    Supports authentication via JWT token or API key passed as query parameters.
    
    Query Parameters:
    - token: JWT authentication token
    - api_key: API key for authentication
    
    Message Format:
    All messages should be JSON with the following structure:
    {
        "type": "message_type",
        "id": "optional_message_id",
        "data": { ... }
    }
    
    Supported message types:
    - ping: Heartbeat message
    - authenticate: Authenticate the connection
    - subscribe: Subscribe to a channel
    - unsubscribe: Unsubscribe from a channel
    - get_channels: Get available channels
    - get_history: Get channel message history
    - get_status: Get connection status
    """
    connection_id = await websocket_manager.connect(websocket)
    
    try:
        # Attempt authentication if credentials provided
        if token or api_key:
            auth_result = await websocket_manager.authenticator.authenticate_connection(
                connection_id, token, api_key
            )
            if auth_result:
                logger.info(f"WebSocket connection {connection_id} authenticated for user {auth_result['user_id']}")
            else:
                logger.warning(f"WebSocket connection {connection_id} authentication failed")
        
        # Message handling loop
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                
                # Handle message
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {connection_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error handling message from {connection_id}: {e}")
                # Send error response
                error_response = {
                    "type": "error",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "error": {
                        "message": "Internal server error",
                        "code": "INTERNAL_ERROR"
                    }
                }
                try:
                    await websocket.send_json(error_response)
                except:
                    break
    
    except Exception as e:
        logger.error(f"WebSocket connection error for {connection_id}: {e}")
    
    finally:
        # Clean up connection
        await websocket_manager.disconnect(connection_id)


@router.get("/ws/stats")
async def get_websocket_stats(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Get WebSocket server statistics.
    
    Returns comprehensive statistics about the WebSocket server including:
    - Server uptime and connection counts
    - Authentication statistics
    - Channel statistics
    - Handler statistics
    """
    return websocket_manager.get_statistics()


@router.get("/ws/connections")
async def list_websocket_connections(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    List active WebSocket connections.
    
    Returns information about all active WebSocket connections including
    connection details, authentication status, and subscribed channels.
    """
    connections = []
    
    for connection_id in websocket_manager.connections:
        connection_info = websocket_manager.get_connection_info(connection_id)
        if connection_info:
            connections.append(connection_info)
    
    return {
        "total_connections": len(connections),
        "connections": connections
    }


@router.get("/ws/channels")
async def list_websocket_channels(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    List WebSocket channels.
    
    Returns information about all available WebSocket channels including
    subscriber counts, metadata, and recent activity.
    """
    channels = websocket_manager.channel_manager.list_channels()
    
    return {
        "total_channels": len(channels),
        "channels": channels
    }


@router.post("/ws/broadcast")
async def broadcast_message(
    message_data: dict,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Broadcast a message to WebSocket clients.
    
    This endpoint allows server-side components to broadcast messages
    to connected WebSocket clients.
    
    Request body should contain:
    - event_type: Type of event to broadcast
    - channel: Target channel (optional)
    - user_id: Target user ID (optional)
    - data: Message payload
    """
    from ..websocket.events import WebSocketEvent, EventType
    from datetime import datetime
    
    try:
        event_type = EventType(message_data.get("event_type", "user_notification"))
        
        event = WebSocketEvent(
            event_type=event_type,
            event_id=f"broadcast_{int(datetime.utcnow().timestamp())}",
            user_id=message_data.get("user_id"),
            channel=message_data.get("channel"),
            data=message_data.get("data", {}),
            metadata=message_data.get("metadata", {})
        )
        
        sent_count = await websocket_manager.broadcast_event(event)
        
        return {
            "success": True,
            "event_id": event.event_id,
            "sent_to_connections": sent_count,
            "timestamp": event.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/ws/notify")
async def send_notification(
    notification_data: dict,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Send a notification to a specific user.
    
    Request body should contain:
    - user_id: Target user ID
    - title: Notification title
    - message: Notification message
    - type: Notification type (optional, default: "info")
    - priority: Notification priority (optional, default: "normal")
    - action_url: Action URL (optional)
    """
    try:
        success = await websocket_manager.handler.send_notification(
            user_id=notification_data["user_id"],
            title=notification_data["title"],
            message=notification_data["message"],
            notification_type=notification_data.get("type", "info"),
            priority=notification_data.get("priority", "normal"),
            action_url=notification_data.get("action_url")
        )
        
        return {
            "success": success,
            "message": "Notification sent" if success else "Failed to send notification"
        }
        
    except KeyError as e:
        return {
            "success": False,
            "error": f"Missing required field: {e}"
        }
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        return {
            "success": False,
            "error": str(e)
        }