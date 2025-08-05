"""
Advanced abuse prevention system for Project Synapse.

Provides intelligent abuse detection, behavioral analysis,
and automated response mechanisms.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics

from fastapi import Request, Response
from ..logging_config import get_logger
from ..metrics_collector import get_metrics_collector


class AbuseType(str, Enum):
    """Types of abuse patterns."""
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    CONTENT_SCRAPING = "content_scraping"
    API_ABUSE = "api_abuse"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CREDENTIAL_STUFFING = "credential_stuffing"
    DDOS_ATTEMPT = "ddos_attempt"
    BOT_ACTIVITY = "bot_activity"


class ActionType(str, Enum):
    """Types of actions to take against abuse."""
    LOG = "log"
    WARN = "warn"
    THROTTLE = "throttle"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    CAPTCHA_CHALLENGE = "captcha_challenge"
    ACCOUNT_SUSPENSION = "account_suspension"


@dataclass
class AbuseEvent:
    """Represents an abuse event."""
    timestamp: datetime
    abuse_type: AbuseType
    severity: int  # 1-10 scale
    client_id: str
    details: Dict[str, Any]
    action_taken: Optional[ActionType] = None


@dataclass
class ClientBehavior:
    """Tracks behavior patterns for a client."""
    client_id: str
    first_seen: datetime
    last_seen: datetime
    request_count: int = 0
    error_count: int = 0
    abuse_events: List[AbuseEvent] = field(default_factory=list)
    request_intervals: deque = field(default_factory=lambda: deque(maxlen=100))
    endpoints_accessed: Set[str] = field(default_factory=set)
    user_agents: Set[str] = field(default_factory=set)
    response_times: deque = field(default_factory=lambda: deque(maxlen=50))
    payload_sizes: deque = field(default_factory=lambda: deque(maxlen=50))
    status_codes: defaultdict = field(default_factory=lambda: defaultdict(int))
    
    def add_request(self, request: Request, response_time: float, 
                   status_code: int, payload_size: int):
        """Add a request to the behavior tracking."""
        now = datetime.utcnow()
        
        # Update basic stats
        if self.last_seen:
            interval = (now - self.last_seen).total_seconds()
            self.request_intervals.append(interval)
        
        self.last_seen = now
        self.request_count += 1
        
        if status_code >= 400:
            self.error_count += 1
        
        # Track patterns
        self.endpoints_accessed.add(str(request.url.path))
        user_agent = request.headers.get('user-agent', 'unknown')
        self.user_agents.add(user_agent)
        self.response_times.append(response_time)
        self.payload_sizes.append(payload_size)
        self.status_codes[status_code] += 1
    
    def get_request_rate(self, window_seconds: int = 60) -> float:
        """Get request rate over specified window."""
        if not self.request_intervals:
            return 0.0
        
        recent_intervals = [
            interval for interval in self.request_intervals
            if interval <= window_seconds
        ]
        
        if not recent_intervals:
            return 0.0
        
        return len(recent_intervals) / window_seconds
    
    def get_error_rate(self) -> float:
        """Get error rate percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    def get_behavior_score(self) -> float:
        """Calculate behavior score (0-100, higher = more suspicious)."""
        score = 0.0
        
        # High request rate
        request_rate = self.get_request_rate()
        if request_rate > 10:  # More than 10 requests per second
            score += min(30, request_rate * 2)
        
        # High error rate
        error_rate = self.get_error_rate()
        if error_rate > 20:  # More than 20% errors
            score += min(25, error_rate)
        
        # Too many different endpoints
        if len(self.endpoints_accessed) > 50:
            score += min(20, len(self.endpoints_accessed) / 5)
        
        # Multiple user agents (possible bot rotation)
        if len(self.user_agents) > 5:
            score += min(15, len(self.user_agents) * 2)
        
        # Consistent timing (bot-like behavior)
        if len(self.request_intervals) > 10:
            interval_variance = statistics.variance(list(self.request_intervals))
            if interval_variance < 0.1:  # Very consistent timing
                score += 10
        
        return min(100, score)


@dataclass
class AbuseRule:
    """Defines an abuse detection rule."""
    name: str
    abuse_type: AbuseType
    description: str
    condition: callable  # Function that takes ClientBehavior and returns bool
    severity: int  # 1-10
    action: ActionType
    cooldown_seconds: int = 300  # Time before rule can trigger again
    
    def __post_init__(self):
        self.last_triggered: Dict[str, datetime] = {}
    
    def can_trigger(self, client_id: str) -> bool:
        """Check if rule can trigger for client (respects cooldown)."""
        if client_id not in self.last_triggered:
            return True
        
        time_since_last = datetime.utcnow() - self.last_triggered[client_id]
        return time_since_last.total_seconds() >= self.cooldown_seconds
    
    def trigger(self, client_id: str):
        """Mark rule as triggered for client."""
        self.last_triggered[client_id] = datetime.utcnow()


class AbusePreventionSystem:
    """Main abuse prevention system."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'abuse_prevention')
        self.metrics = get_metrics_collector()
        
        # Client behavior tracking
        self.client_behaviors: Dict[str, ClientBehavior] = {}
        
        # Blocked clients
        self.blocked_clients: Dict[str, datetime] = {}  # client_id -> block_expiry
        self.permanently_blocked: Set[str] = set()
        
        # Abuse rules
        self.rules = self._create_default_rules()
        
        # Statistics
        self.stats = {
            'total_requests_analyzed': 0,
            'abuse_events_detected': 0,
            'clients_blocked': 0,
            'false_positives': 0,
            'rules_triggered': defaultdict(int)
        }
    
    def _create_default_rules(self) -> List[AbuseRule]:
        """Create default abuse detection rules."""
        return [
            AbuseRule(
                name="high_request_rate",
                abuse_type=AbuseType.RATE_LIMIT_VIOLATION,
                description="Extremely high request rate",
                condition=lambda behavior: behavior.get_request_rate() > 20,
                severity=8,
                action=ActionType.TEMPORARY_BLOCK,
                cooldown_seconds=600
            ),
            
            AbuseRule(
                name="high_error_rate",
                abuse_type=AbuseType.SUSPICIOUS_BEHAVIOR,
                description="High error rate indicating probing",
                condition=lambda behavior: (
                    behavior.get_error_rate() > 50 and behavior.request_count > 20
                ),
                severity=7,
                action=ActionType.TEMPORARY_BLOCK,
                cooldown_seconds=300
            ),
            
            AbuseRule(
                name="endpoint_enumeration",
                abuse_type=AbuseType.SUSPICIOUS_BEHAVIOR,
                description="Accessing too many different endpoints",
                condition=lambda behavior: len(behavior.endpoints_accessed) > 100,
                severity=6,
                action=ActionType.THROTTLE,
                cooldown_seconds=180
            ),
            
            AbuseRule(
                name="bot_like_timing",
                abuse_type=AbuseType.BOT_ACTIVITY,
                description="Consistent request timing indicating bot",
                condition=lambda behavior: (
                    len(behavior.request_intervals) > 20 and
                    len(behavior.user_agents) == 1 and
                    statistics.variance(list(behavior.request_intervals)) < 0.05
                ),
                severity=5,
                action=ActionType.CAPTCHA_CHALLENGE,
                cooldown_seconds=900
            ),
            
            AbuseRule(
                name="user_agent_rotation",
                abuse_type=AbuseType.BOT_ACTIVITY,
                description="Rotating user agents to avoid detection",
                condition=lambda behavior: len(behavior.user_agents) > 10,
                severity=6,
                action=ActionType.TEMPORARY_BLOCK,
                cooldown_seconds=600
            ),
            
            AbuseRule(
                name="content_scraping_pattern",
                abuse_type=AbuseType.CONTENT_SCRAPING,
                description="Pattern indicating content scraping",
                condition=lambda behavior: (
                    behavior.request_count > 500 and
                    len([ep for ep in behavior.endpoints_accessed 
                         if '/api/content' in ep]) > 50
                ),
                severity=7,
                action=ActionType.TEMPORARY_BLOCK,
                cooldown_seconds=1800
            ),
            
            AbuseRule(
                name="credential_stuffing",
                abuse_type=AbuseType.CREDENTIAL_STUFFING,
                description="Multiple failed authentication attempts",
                condition=lambda behavior: (
                    behavior.status_codes.get(401, 0) > 20 and
                    len([ep for ep in behavior.endpoints_accessed 
                         if '/auth' in ep]) > 0
                ),
                severity=9,
                action=ActionType.TEMPORARY_BLOCK,
                cooldown_seconds=3600
            ),
            
            AbuseRule(
                name="resource_exhaustion",
                abuse_type=AbuseType.RESOURCE_EXHAUSTION,
                description="Requests designed to exhaust resources",
                condition=lambda behavior: (
                    len(behavior.payload_sizes) > 10 and
                    statistics.mean(behavior.payload_sizes) > 1024 * 1024 and  # 1MB average
                    behavior.get_request_rate() > 5
                ),
                severity=8,
                action=ActionType.TEMPORARY_BLOCK,
                cooldown_seconds=1200
            )
        ]
    
    async def analyze_request(self, request: Request, response_time: float,
                             status_code: int, payload_size: int,
                             client_id: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Analyze request for abuse patterns."""
        self.stats['total_requests_analyzed'] += 1
        
        # Generate client ID if not provided
        if not client_id:
            client_id = self._generate_client_id(request)
        
        # Check if client is blocked
        if await self._is_client_blocked(client_id):
            return False, {
                'reason': 'client_blocked',
                'message': 'Client is currently blocked due to abuse',
                'client_id': client_id
            }
        
        # Get or create client behavior
        behavior = self._get_client_behavior(client_id, request)
        
        # Update behavior with current request
        behavior.add_request(request, response_time, status_code, payload_size)
        
        # Check abuse rules
        triggered_rules = []
        for rule in self.rules:
            if rule.can_trigger(client_id) and rule.condition(behavior):
                triggered_rules.append(rule)
                rule.trigger(client_id)
                self.stats['rules_triggered'][rule.name] += 1
        
        # Process triggered rules
        if triggered_rules:
            await self._process_abuse_detection(client_id, behavior, triggered_rules)
            
            # Determine if request should be blocked
            max_severity = max(rule.severity for rule in triggered_rules)
            should_block = any(
                rule.action in [ActionType.TEMPORARY_BLOCK, ActionType.PERMANENT_BLOCK]
                for rule in triggered_rules
            )
            
            if should_block:
                return False, {
                    'reason': 'abuse_detected',
                    'message': 'Request blocked due to abuse detection',
                    'rules_triggered': [rule.name for rule in triggered_rules],
                    'severity': max_severity,
                    'client_id': client_id
                }
            else:
                # Log but allow request
                return True, {
                    'warning': 'abuse_warning',
                    'message': 'Suspicious behavior detected',
                    'rules_triggered': [rule.name for rule in triggered_rules],
                    'client_id': client_id
                }
        
        return True, None
    
    async def _process_abuse_detection(self, client_id: str, behavior: ClientBehavior,
                                      triggered_rules: List[AbuseRule]):
        """Process detected abuse and take appropriate actions."""
        self.stats['abuse_events_detected'] += 1
        
        # Create abuse events
        for rule in triggered_rules:
            event = AbuseEvent(
                timestamp=datetime.utcnow(),
                abuse_type=rule.abuse_type,
                severity=rule.severity,
                client_id=client_id,
                details={
                    'rule_name': rule.name,
                    'rule_description': rule.description,
                    'behavior_score': behavior.get_behavior_score(),
                    'request_count': behavior.request_count,
                    'error_rate': behavior.get_error_rate(),
                    'request_rate': behavior.get_request_rate()
                },
                action_taken=rule.action
            )
            
            behavior.abuse_events.append(event)
            
            # Log abuse event
            self.logger.warning(
                f"Abuse detected: {rule.name}",
                operation="abuse_detection",
                client_id=client_id,
                abuse_type=rule.abuse_type.value,
                severity=rule.severity,
                action=rule.action.value,
                behavior_score=behavior.get_behavior_score()
            )
            
            # Take action
            await self._take_action(client_id, rule.action, rule.severity)
            
            # Record metrics
            counter = self.metrics.get_counter('abuse_events_total')
            counter.increment(1, abuse_type=rule.abuse_type.value, action=rule.action.value, severity=str(rule.severity))
    
    async def _take_action(self, client_id: str, action: ActionType, severity: int):
        """Take action against abusive client."""
        if action == ActionType.TEMPORARY_BLOCK:
            # Block duration based on severity
            block_duration = min(3600, 60 * severity)  # Max 1 hour
            block_expiry = datetime.utcnow() + timedelta(seconds=block_duration)
            self.blocked_clients[client_id] = block_expiry
            self.stats['clients_blocked'] += 1
            
            self.logger.info(
                f"Client temporarily blocked",
                operation="client_block",
                client_id=client_id,
                duration_seconds=block_duration,
                severity=severity
            )
        
        elif action == ActionType.PERMANENT_BLOCK:
            self.permanently_blocked.add(client_id)
            self.stats['clients_blocked'] += 1
            
            self.logger.warning(
                f"Client permanently blocked",
                operation="client_permanent_block",
                client_id=client_id,
                severity=severity
            )
        
        elif action == ActionType.THROTTLE:
            # Throttling is handled by rate limiter
            pass
        
        elif action == ActionType.CAPTCHA_CHALLENGE:
            # Would integrate with CAPTCHA system
            self.logger.info(
                f"CAPTCHA challenge issued",
                operation="captcha_challenge",
                client_id=client_id
            )
    
    async def _is_client_blocked(self, client_id: str) -> bool:
        """Check if client is currently blocked."""
        # Check permanent blocks
        if client_id in self.permanently_blocked:
            return True
        
        # Check temporary blocks
        if client_id in self.blocked_clients:
            block_expiry = self.blocked_clients[client_id]
            if datetime.utcnow() < block_expiry:
                return True
            else:
                # Block expired, remove it
                del self.blocked_clients[client_id]
        
        return False
    
    def _get_client_behavior(self, client_id: str, request: Request) -> ClientBehavior:
        """Get or create client behavior tracking."""
        if client_id not in self.client_behaviors:
            self.client_behaviors[client_id] = ClientBehavior(
                client_id=client_id,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
        
        return self.client_behaviors[client_id]
    
    def _generate_client_id(self, request: Request) -> str:
        """Generate client ID from request."""
        # Use IP address and User-Agent for identification
        client_ip = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get('user-agent', 'unknown')
        
        # Create hash for privacy
        client_data = f"{client_ip}:{user_agent}"
        return hashlib.sha256(client_data.encode()).hexdigest()[:16]
    
    def add_custom_rule(self, rule: AbuseRule):
        """Add a custom abuse detection rule."""
        self.rules.append(rule)
        self.logger.info(f"Added custom abuse rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an abuse detection rule."""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        self.logger.info(f"Removed abuse rule: {rule_name}")
    
    def whitelist_client(self, client_id: str):
        """Whitelist a client (remove from blocks)."""
        if client_id in self.blocked_clients:
            del self.blocked_clients[client_id]
        
        if client_id in self.permanently_blocked:
            self.permanently_blocked.remove(client_id)
        
        self.logger.info(f"Client whitelisted: {client_id}")
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a client."""
        if client_id not in self.client_behaviors:
            return None
        
        behavior = self.client_behaviors[client_id]
        
        return {
            'client_id': client_id,
            'first_seen': behavior.first_seen.isoformat(),
            'last_seen': behavior.last_seen.isoformat(),
            'request_count': behavior.request_count,
            'error_count': behavior.error_count,
            'error_rate': behavior.get_error_rate(),
            'request_rate': behavior.get_request_rate(),
            'behavior_score': behavior.get_behavior_score(),
            'endpoints_accessed': len(behavior.endpoints_accessed),
            'user_agents': len(behavior.user_agents),
            'abuse_events': len(behavior.abuse_events),
            'is_blocked': client_id in self.blocked_clients,
            'block_expiry': (
                self.blocked_clients[client_id].isoformat()
                if client_id in self.blocked_clients else None
            ),
            'is_permanently_blocked': client_id in self.permanently_blocked
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get abuse prevention system statistics."""
        return {
            **self.stats,
            'active_clients': len(self.client_behaviors),
            'blocked_clients': len(self.blocked_clients),
            'permanently_blocked_clients': len(self.permanently_blocked),
            'rules_configured': len(self.rules),
            'abuse_detection_rate': (
                self.stats['abuse_events_detected'] / 
                max(1, self.stats['total_requests_analyzed'])
            )
        }
    
    async def cleanup_expired_data(self):
        """Clean up expired data and blocks."""
        current_time = datetime.utcnow()
        
        # Clean up expired blocks
        expired_blocks = [
            client_id for client_id, expiry in self.blocked_clients.items()
            if current_time >= expiry
        ]
        
        for client_id in expired_blocks:
            del self.blocked_clients[client_id]
        
        # Clean up old client behaviors (older than 7 days)
        cutoff_time = current_time - timedelta(days=7)
        expired_behaviors = [
            client_id for client_id, behavior in self.client_behaviors.items()
            if behavior.last_seen < cutoff_time
        ]
        
        for client_id in expired_behaviors:
            del self.client_behaviors[client_id]
        
        if expired_blocks or expired_behaviors:
            self.logger.info(
                f"Cleaned up {len(expired_blocks)} expired blocks and "
                f"{len(expired_behaviors)} old client behaviors"
            )
        
        return len(expired_blocks) + len(expired_behaviors)


# Global instance
_abuse_prevention_system: Optional[AbusePreventionSystem] = None


def get_abuse_prevention_system() -> AbusePreventionSystem:
    """Get the global abuse prevention system instance."""
    global _abuse_prevention_system
    if _abuse_prevention_system is None:
        _abuse_prevention_system = AbusePreventionSystem()
    return _abuse_prevention_system


async def start_abuse_prevention_cleanup_task():
    """Start background cleanup task for abuse prevention system."""
    system = get_abuse_prevention_system()
    
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await system.cleanup_expired_data()
        except Exception as e:
            logger = get_logger(__name__, 'abuse_prevention_cleanup')
            logger.error(f"Error in abuse prevention cleanup: {e}")