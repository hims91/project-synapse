"""
Webhook delivery system.

Handles reliable webhook delivery with retry logic, exponential backoff,
and comprehensive error handling.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import aiohttp
from aiohttp import ClientTimeout, ClientError

from .models import (
    WebhookEndpoint, WebhookEvent, WebhookDelivery, WebhookDeliveryAttempt,
    DeliveryStatus, WebhookEventType
)
from .security import WebhookSecurity
from .validation import WebhookValidator

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """Manages webhook delivery with retry logic and error handling."""
    
    def __init__(self, max_concurrent_deliveries: int = 10):
        self.max_concurrent_deliveries = max_concurrent_deliveries
        self.delivery_semaphore = asyncio.Semaphore(max_concurrent_deliveries)
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.active_deliveries: Dict[str, WebhookDelivery] = {}
        self.stats = {
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'retries_attempted': 0,
            'average_delivery_time_ms': 0
        }
        
        # Start background workers
        self._workers_started = False
    
    async def start_workers(self):
        """Start background delivery workers."""
        if self._workers_started:
            return
        
        self._workers_started = True
        
        # Start delivery workers
        for i in range(3):  # 3 concurrent workers
            asyncio.create_task(self._delivery_worker(f"worker-{i}"))
        
        # Start retry scheduler
        asyncio.create_task(self._retry_scheduler())
        
        logger.info("Webhook delivery workers started")
    
    async def deliver_event(
        self,
        event: WebhookEvent,
        endpoints: List[WebhookEndpoint]
    ) -> List[WebhookDelivery]:
        """
        Deliver event to multiple endpoints.
        
        Args:
            event: Event to deliver
            endpoints: List of webhook endpoints
            
        Returns:
            List of delivery records
        """
        deliveries = []
        
        for endpoint in endpoints:
            # Check if endpoint should receive this event
            if not endpoint.matches_event(event.event_type, event.data):
                continue
            
            # Create delivery record
            delivery = WebhookDelivery(
                endpoint_id=endpoint.id,
                event_id=event.id,
                max_attempts=endpoint.max_retries + 1,  # +1 for initial attempt
                scheduled_at=datetime.utcnow()
            )
            
            deliveries.append(delivery)
            
            # Queue for delivery
            await self.delivery_queue.put((delivery, endpoint, event))
        
        return deliveries
    
    async def _delivery_worker(self, worker_id: str):
        """Background worker for processing delivery queue."""
        logger.info(f"Delivery worker {worker_id} started")
        
        while True:
            try:
                # Get delivery from queue
                delivery, endpoint, event = await self.delivery_queue.get()
                
                # Process delivery with semaphore to limit concurrency
                async with self.delivery_semaphore:
                    await self._process_delivery(delivery, endpoint, event)
                
                self.delivery_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in delivery worker {worker_id}: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _process_delivery(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint,
        event: WebhookEvent
    ):
        """Process a single webhook delivery."""
        delivery_id = str(delivery.id)
        self.active_deliveries[delivery_id] = delivery
        
        try:
            # Update delivery status
            delivery.status = DeliveryStatus.SENDING
            delivery.started_at = datetime.utcnow()
            delivery.attempt_count += 1
            
            # Create delivery attempt
            attempt = await self._attempt_delivery(delivery, endpoint, event)
            
            # Update delivery based on attempt result
            if attempt.is_success():
                delivery.status = DeliveryStatus.SUCCESS
                delivery.completed_at = datetime.utcnow()
                delivery.response_status_code = attempt.response_status_code
                delivery.response_headers = attempt.response_headers
                delivery.response_body = attempt.response_body
                
                # Update endpoint stats
                endpoint.update_stats(success=True)
                
                # Update service stats
                self.stats['successful_deliveries'] += 1
                
                logger.info(f"Webhook delivered successfully: {delivery_id}")
                
            else:
                # Delivery failed
                delivery.status = DeliveryStatus.FAILED
                delivery.completed_at = datetime.utcnow()
                delivery.response_status_code = attempt.response_status_code
                delivery.response_headers = attempt.response_headers
                delivery.response_body = attempt.response_body
                delivery.error_message = attempt.error_message
                
                # Check if we should retry
                if delivery.attempt_count < delivery.max_attempts:
                    # Schedule retry
                    delivery.status = DeliveryStatus.RETRYING
                    delivery.next_retry_at = delivery.calculate_next_retry(
                        endpoint.retry_delay_seconds
                    )
                    delivery.completed_at = None
                    
                    logger.info(
                        f"Webhook delivery failed, scheduling retry {delivery.attempt_count}/{delivery.max_attempts}: {delivery_id}"
                    )
                    
                    # Re-queue for retry
                    await self.delivery_queue.put((delivery, endpoint, event))
                    
                else:
                    # Max attempts reached
                    delivery.status = DeliveryStatus.ABANDONED
                    
                    # Update endpoint stats
                    endpoint.update_stats(success=False)
                    
                    # Update service stats
                    self.stats['failed_deliveries'] += 1
                    
                    logger.error(
                        f"Webhook delivery abandoned after {delivery.attempt_count} attempts: {delivery_id}"
                    )
            
            # Update total deliveries
            self.stats['total_deliveries'] += 1
            
            # Update average delivery time
            if delivery.duration_ms:
                current_avg = self.stats['average_delivery_time_ms']
                total_deliveries = self.stats['total_deliveries']
                self.stats['average_delivery_time_ms'] = (
                    (current_avg * (total_deliveries - 1) + delivery.duration_ms) / total_deliveries
                )
        
        except Exception as e:
            logger.error(f"Error processing delivery {delivery_id}: {e}")
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            delivery.completed_at = datetime.utcnow()
        
        finally:
            # Remove from active deliveries
            self.active_deliveries.pop(delivery_id, None)
    
    async def _attempt_delivery(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint,
        event: WebhookEvent
    ) -> WebhookDeliveryAttempt:
        """Attempt to deliver webhook to endpoint."""
        attempt = WebhookDeliveryAttempt(
            attempt_number=delivery.attempt_count,
            started_at=datetime.utcnow(),
            request_url=str(endpoint.url),
            timeout_seconds=endpoint.timeout_seconds
        )
        
        try:
            # Prepare payload
            payload = event.to_webhook_payload()
            payload_json = json.dumps(payload, default=str)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Synapse-Webhook/1.0',
                'X-Synapse-Event': event.event_type,
                'X-Synapse-Event-ID': str(event.id),
                'X-Synapse-Delivery-ID': WebhookSecurity.create_delivery_id(),
                'X-Synapse-Timestamp': str(int(event.timestamp.timestamp()))
            }
            
            # Add custom headers
            headers.update(WebhookSecurity.sanitize_headers(endpoint.headers))
            
            # Add signature headers if secret is configured
            if endpoint.secret:
                signature_headers = WebhookSecurity.create_signature_headers(
                    payload_json,
                    endpoint.secret,
                    headers['X-Synapse-Timestamp']
                )
                headers.update(signature_headers)
            
            # Store request details
            attempt.request_headers = headers
            attempt.request_body = payload_json
            
            # Make HTTP request
            timeout = ClientTimeout(total=endpoint.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = datetime.utcnow()
                
                async with session.post(
                    str(endpoint.url),
                    data=payload_json,
                    headers=headers,
                    allow_redirects=False  # Don't follow redirects for security
                ) as response:
                    end_time = datetime.utcnow()
                    
                    # Read response
                    response_body = await response.text()
                    
                    # Update attempt with response details
                    attempt.completed_at = end_time
                    attempt.response_status_code = response.status
                    attempt.response_headers = dict(response.headers)
                    attempt.response_body = response_body[:1000]  # Limit size
                    attempt.duration_ms = int((end_time - start_time).total_seconds() * 1000)
                    
                    # Validate response
                    is_success, error_msg = WebhookValidator.validate_webhook_response(
                        response.status,
                        dict(response.headers),
                        response_body
                    )
                    
                    if not is_success:
                        attempt.error_message = error_msg
        
        except asyncio.TimeoutError:
            attempt.completed_at = datetime.utcnow()
            attempt.error_type = "timeout"
            attempt.error_message = f"Request timeout after {endpoint.timeout_seconds} seconds"
            
        except ClientError as e:
            attempt.completed_at = datetime.utcnow()
            attempt.error_type = "client_error"
            attempt.error_message = f"HTTP client error: {str(e)}"
            
        except Exception as e:
            attempt.completed_at = datetime.utcnow()
            attempt.error_type = "unknown_error"
            attempt.error_message = f"Unexpected error: {str(e)}"
        
        return attempt
    
    async def _retry_scheduler(self):
        """Background scheduler for processing retries."""
        logger.info("Retry scheduler started")
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Find deliveries ready for retry
                current_time = datetime.utcnow()
                
                for delivery in list(self.active_deliveries.values()):
                    if (delivery.status == DeliveryStatus.RETRYING and
                        delivery.next_retry_at and
                        delivery.next_retry_at <= current_time):
                        
                        # This would need to be implemented with proper storage
                        # For now, we rely on the queue-based retry mechanism
                        pass
                
            except Exception as e:
                logger.error(f"Error in retry scheduler: {e}")
    
    async def test_endpoint(self, endpoint: WebhookEndpoint) -> Tuple[bool, str]:
        """
        Test webhook endpoint with a test event.
        
        Args:
            endpoint: Webhook endpoint to test
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Create test event
            test_event = WebhookEvent(
                event_type=WebhookEventType.SYSTEM_HEALTH,
                data={
                    "test": True,
                    "message": "This is a test webhook delivery",
                    "timestamp": datetime.utcnow().isoformat()
                },
                metadata={
                    "category": "test",
                    "source_component": "webhook_delivery_service"
                }
            )
            
            # Create test delivery
            delivery = WebhookDelivery(
                endpoint_id=endpoint.id,
                event_id=test_event.id,
                max_attempts=1,
                scheduled_at=datetime.utcnow()
            )
            
            # Attempt delivery
            attempt = await self._attempt_delivery(delivery, endpoint, test_event)
            
            if attempt.is_success():
                return True, f"Test successful (HTTP {attempt.response_status_code})"
            else:
                return False, f"Test failed: {attempt.error_message or 'Unknown error'}"
        
        except Exception as e:
            return False, f"Test error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get delivery service statistics."""
        return {
            **self.stats,
            'active_deliveries': len(self.active_deliveries),
            'queue_size': self.delivery_queue.qsize(),
            'max_concurrent_deliveries': self.max_concurrent_deliveries
        }
    
    async def get_delivery_status(self, delivery_id: UUID) -> Optional[WebhookDelivery]:
        """Get status of a specific delivery."""
        return self.active_deliveries.get(str(delivery_id))
    
    async def cancel_delivery(self, delivery_id: UUID) -> bool:
        """Cancel a pending delivery."""
        delivery = self.active_deliveries.get(str(delivery_id))
        if delivery and delivery.status in [DeliveryStatus.PENDING, DeliveryStatus.RETRYING]:
            delivery.status = DeliveryStatus.ABANDONED
            delivery.error_message = "Cancelled by user"
            delivery.completed_at = datetime.utcnow()
            return True
        return False


class WebhookEventBus:
    """Event bus for webhook system."""
    
    def __init__(self, delivery_service: WebhookDeliveryService):
        self.delivery_service = delivery_service
        self.endpoints: Dict[UUID, WebhookEndpoint] = {}
        self.event_history: List[WebhookEvent] = []
        self.max_history_size = 1000
    
    async def register_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Register a webhook endpoint."""
        # Validate endpoint
        is_valid, errors = WebhookValidator.validate_endpoint(endpoint)
        if not is_valid:
            logger.error(f"Invalid webhook endpoint: {errors}")
            return False
        
        self.endpoints[endpoint.id] = endpoint
        logger.info(f"Registered webhook endpoint: {endpoint.name} ({endpoint.id})")
        return True
    
    async def unregister_endpoint(self, endpoint_id: UUID) -> bool:
        """Unregister a webhook endpoint."""
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints.pop(endpoint_id)
            logger.info(f"Unregistered webhook endpoint: {endpoint.name} ({endpoint_id})")
            return True
        return False
    
    async def update_endpoint(self, endpoint: WebhookEndpoint) -> bool:
        """Update a webhook endpoint."""
        # Validate endpoint
        is_valid, errors = WebhookValidator.validate_endpoint(endpoint)
        if not is_valid:
            logger.error(f"Invalid webhook endpoint update: {errors}")
            return False
        
        if endpoint.id in self.endpoints:
            self.endpoints[endpoint.id] = endpoint
            logger.info(f"Updated webhook endpoint: {endpoint.name} ({endpoint.id})")
            return True
        return False
    
    async def publish_event(self, event: WebhookEvent) -> List[WebhookDelivery]:
        """
        Publish event to all matching endpoints.
        
        Args:
            event: Event to publish
            
        Returns:
            List of delivery records
        """
        # Validate event
        is_valid, errors = WebhookValidator.validate_event(event)
        if not is_valid:
            logger.error(f"Invalid webhook event: {errors}")
            return []
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
        
        # Find matching endpoints
        matching_endpoints = [
            endpoint for endpoint in self.endpoints.values()
            if (endpoint.status.value == "active" and
                endpoint.matches_event(event.event_type, event.data))
        ]
        
        if not matching_endpoints:
            logger.debug(f"No matching endpoints for event {event.event_type}")
            return []
        
        # Deliver to matching endpoints
        deliveries = await self.delivery_service.deliver_event(event, matching_endpoints)
        
        logger.info(
            f"Published event {event.event_type} to {len(matching_endpoints)} endpoints, "
            f"created {len(deliveries)} deliveries"
        )
        
        return deliveries
    
    def get_endpoints(self, user_id: Optional[str] = None) -> List[WebhookEndpoint]:
        """Get webhook endpoints, optionally filtered by user."""
        endpoints = list(self.endpoints.values())
        
        if user_id:
            endpoints = [ep for ep in endpoints if ep.user_id == user_id]
        
        return endpoints
    
    def get_endpoint(self, endpoint_id: UUID) -> Optional[WebhookEndpoint]:
        """Get specific webhook endpoint."""
        return self.endpoints.get(endpoint_id)
    
    def get_recent_events(self, limit: int = 50) -> List[WebhookEvent]:
        """Get recent events."""
        return self.event_history[-limit:]


# Global instances
webhook_delivery_service = WebhookDeliveryService()
webhook_event_bus = WebhookEventBus(webhook_delivery_service)


async def start_webhook_system():
    """Start the webhook delivery system."""
    await webhook_delivery_service.start_workers()
    logger.info("Webhook system started")


def get_webhook_delivery_service() -> WebhookDeliveryService:
    """Get the webhook delivery service instance."""
    return webhook_delivery_service


def get_webhook_event_bus() -> WebhookEventBus:
    """Get the webhook event bus instance."""
    return webhook_event_bus