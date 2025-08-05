"""
Alerting system for Project Synapse.

Provides comprehensive alerting with escalation policies, multiple notification
channels, and intelligent alert management.
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .logging_config import get_logger
from .metrics_collector import MetricsCollector, MetricSeries


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"


@dataclass
class AlertRule:
    """Alert rule definition."""
    id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 80", "< 0.95", "== 0"
    threshold: Union[int, float]
    severity: AlertSeverity
    duration_minutes: int = 5  # How long condition must be true
    evaluation_interval: int = 60  # Seconds between evaluations
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    
    # Suppression settings
    suppress_duration_minutes: int = 60
    max_alerts_per_hour: int = 10
    
    def evaluate(self, metric_value: Union[int, float]) -> bool:
        """Evaluate if the alert condition is met."""
        if not self.enabled:
            return False
        
        try:
            if self.condition.startswith('>='):
                return metric_value >= float(self.condition[2:].strip())
            elif self.condition.startswith('<='):
                return metric_value <= float(self.condition[2:].strip())
            elif self.condition.startswith('>'):
                return metric_value > float(self.condition[1:].strip())
            elif self.condition.startswith('<'):
                return metric_value < float(self.condition[1:].strip())
            elif self.condition.startswith('=='):
                return metric_value == float(self.condition[2:].strip())
            elif self.condition.startswith('!='):
                return metric_value != float(self.condition[2:].strip())
            else:
                # Default to greater than
                return metric_value > self.threshold
        except (ValueError, TypeError):
            return False


@dataclass
class Alert:
    """An active alert."""
    id: str
    rule_id: str
    rule_name: str
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    severity: AlertSeverity
    status: AlertStatus
    message: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    escalation_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.utcnow()
        return end_time - self.created_at
    
    def is_active(self) -> bool:
        """Check if alert is active."""
        return self.status == AlertStatus.ACTIVE
    
    def acknowledge(self, acknowledged_by: str):
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = acknowledged_by
        self.updated_at = datetime.utcnow()
    
    def resolve(self):
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def suppress(self):
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.utcnow()


@dataclass
class EscalationPolicy:
    """Escalation policy for alerts."""
    id: str
    name: str
    description: str
    levels: List[Dict[str, Any]]  # Each level has channels and delay
    
    def get_escalation_level(self, alert_duration_minutes: int) -> int:
        """Get current escalation level based on alert duration."""
        for i, level in enumerate(self.levels):
            if alert_duration_minutes < level.get('delay_minutes', 0):
                return max(0, i - 1)
        return len(self.levels) - 1


@dataclass
class NotificationChannelConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    
    # Rate limiting
    max_notifications_per_hour: int = 60
    cooldown_minutes: int = 5


class NotificationSender:
    """Handles sending notifications through various channels."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'notification_sender')
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting tracking
        self.notification_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_notification: Dict[str, datetime] = {}
    
    async def start(self):
        """Start the notification sender."""
        self.session = aiohttp.ClientSession()
        self.logger.info("Notification sender started", operation="start_sender")
    
    async def stop(self):
        """Stop the notification sender."""
        if self.session:
            await self.session.close()
        self.logger.info("Notification sender stopped", operation="stop_sender")
    
    async def send_notification(
        self,
        alert: Alert,
        channel_config: NotificationChannelConfig
    ) -> bool:
        """Send notification through specified channel."""
        if not channel_config.enabled:
            return False
        
        # Check rate limiting
        if not self._check_rate_limit(alert, channel_config):
            self.logger.warning(
                f"Rate limit exceeded for {channel_config.channel}",
                operation="send_notification",
                alert_id=alert.id,
                channel=channel_config.channel.value
            )
            return False
        
        try:
            success = False
            
            if channel_config.channel == NotificationChannel.EMAIL:
                success = await self._send_email(alert, channel_config.config)
            elif channel_config.channel == NotificationChannel.SLACK:
                success = await self._send_slack(alert, channel_config.config)
            elif channel_config.channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(alert, channel_config.config)
            elif channel_config.channel == NotificationChannel.DISCORD:
                success = await self._send_discord(alert, channel_config.config)
            
            if success:
                # Track notification
                self._track_notification(alert, channel_config)
                
                self.logger.info(
                    f"Notification sent via {channel_config.channel}",
                    operation="send_notification",
                    alert_id=alert.id,
                    channel=channel_config.channel.value
                )
            
            return success
        
        except Exception as e:
            self.logger.error(
                f"Error sending notification via {channel_config.channel}",
                operation="send_notification",
                alert_id=alert.id,
                channel=channel_config.channel.value,
                error=str(e)
            )
            return False
    
    def _check_rate_limit(self, alert: Alert, channel_config: NotificationChannelConfig) -> bool:
        """Check if notification is within rate limits."""
        key = f"{alert.rule_id}:{channel_config.channel.value}"
        now = datetime.utcnow()
        
        # Check cooldown
        if key in self.last_notification:
            time_since_last = (now - self.last_notification[key]).total_seconds() / 60
            if time_since_last < channel_config.cooldown_minutes:
                return False
        
        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        recent_notifications = [
            ts for ts in self.notification_counts[key]
            if ts > hour_ago
        ]
        
        return len(recent_notifications) < channel_config.max_notifications_per_hour
    
    def _track_notification(self, alert: Alert, channel_config: NotificationChannelConfig):
        """Track notification for rate limiting."""
        key = f"{alert.rule_id}:{channel_config.channel.value}"
        now = datetime.utcnow()
        
        self.notification_counts[key].append(now)
        self.last_notification[key] = now
        
        # Add to alert's notification history
        alert.notifications_sent.append({
            'channel': channel_config.channel.value,
            'timestamp': now.isoformat(),
            'success': True
        })
    
    async def _send_email(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send email notification."""
        try:
            smtp_server = config.get('smtp_server', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            from_email = config.get('from_email')
            to_emails = config.get('to_emails', [])
            
            if not to_emails or not from_email:
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.rule_name}"
            
            # Email body
            body = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.upper()}
Status: {alert.status.upper()}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}
Message: {alert.message}
Created: {alert.created_at.isoformat()}
Duration: {alert.get_duration()}

Alert ID: {alert.id}
Rule ID: {alert.rule_id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                
                server.send_message(msg)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_slack(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send Slack notification."""
        if not self.session:
            return False
        
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return False
        
        # Color based on severity
        color_map = {
            AlertSeverity.LOW: '#36a64f',      # Green
            AlertSeverity.MEDIUM: '#ff9500',   # Orange
            AlertSeverity.HIGH: '#ff0000',     # Red
            AlertSeverity.CRITICAL: '#8b0000'  # Dark Red
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(alert.severity, '#ff0000'),
                'title': f'{alert.severity.upper()} Alert: {alert.rule_name}',
                'text': alert.message,
                'fields': [
                    {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                    {'title': 'Current Value', 'value': str(alert.current_value), 'short': True},
                    {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                    {'title': 'Duration', 'value': str(alert.get_duration()), 'short': True},
                    {'title': 'Alert ID', 'value': alert.id, 'short': False}
                ],
                'timestamp': int(alert.created_at.timestamp())
            }]
        }
        
        try:
            async with self.session.post(webhook_url, json=payload) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _send_webhook(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        if not self.session:
            return False
        
        url = config.get('url')
        if not url:
            return False
        
        headers = config.get('headers', {})
        headers['Content-Type'] = 'application/json'
        
        payload = {
            'alert': alert.to_dict(),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'synapse-alerting'
        }
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                return 200 <= response.status < 300
        except Exception:
            return False
    
    async def _send_discord(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send Discord notification."""
        if not self.session:
            return False
        
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return False
        
        # Color based on severity
        color_map = {
            AlertSeverity.LOW: 0x36a64f,      # Green
            AlertSeverity.MEDIUM: 0xff9500,   # Orange
            AlertSeverity.HIGH: 0xff0000,     # Red
            AlertSeverity.CRITICAL: 0x8b0000  # Dark Red
        }
        
        embed = {
            'title': f'{alert.severity.upper()} Alert: {alert.rule_name}',
            'description': alert.message,
            'color': color_map.get(alert.severity, 0xff0000),
            'fields': [
                {'name': 'Metric', 'value': alert.metric_name, 'inline': True},
                {'name': 'Current Value', 'value': str(alert.current_value), 'inline': True},
                {'name': 'Threshold', 'value': str(alert.threshold), 'inline': True},
                {'name': 'Duration', 'value': str(alert.get_duration()), 'inline': True},
                {'name': 'Alert ID', 'value': alert.id, 'inline': False}
            ],
            'timestamp': alert.created_at.isoformat()
        }
        
        payload = {'embeds': [embed]}
        
        try:
            async with self.session.post(webhook_url, json=payload) as response:
                return response.status == 204
        except Exception:
            return False


class AlertManager:
    """Central alert management system."""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'alert_manager')
        self.metrics_collector = MetricsCollector.get_instance()
        self.notification_sender = NotificationSender()
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Escalation policies
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannelConfig] = {}
        
        # Evaluation state
        self.rule_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Background tasks
        self.evaluation_task: Optional[asyncio.Task] = None
        self.escalation_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'alerts_created': 0,
            'alerts_resolved': 0,
            'notifications_sent': 0,
            'rules_evaluated': 0,
            'start_time': None
        }
        
        # Setup default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                id='high_cpu_usage',
                name='High CPU Usage',
                description='CPU usage is above 80%',
                metric_name='system_cpu_percent',
                condition='> 80',
                threshold=80,
                severity=AlertSeverity.HIGH,
                duration_minutes=5,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                id='high_memory_usage',
                name='High Memory Usage',
                description='Memory usage is above 90%',
                metric_name='system_memory_percent',
                condition='> 90',
                threshold=90,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=3,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                id='high_error_rate',
                name='High Error Rate',
                description='HTTP error rate is above 5%',
                metric_name='http_error_rate',
                condition='> 5',
                threshold=5,
                severity=AlertSeverity.HIGH,
                duration_minutes=2,
                notification_channels=[NotificationChannel.SLACK]
            ),
            AlertRule(
                id='disk_space_low',
                name='Low Disk Space',
                description='Disk usage is above 85%',
                metric_name='system_disk_percent',
                condition='> 85',
                threshold=85,
                severity=AlertSeverity.MEDIUM,
                duration_minutes=10,
                notification_channels=[NotificationChannel.EMAIL]
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    async def start(self):
        """Start the alert manager."""
        if self.running:
            return
        
        self.running = True
        self.stats['start_time'] = datetime.utcnow()
        
        # Start notification sender
        await self.notification_sender.start()
        
        # Start background tasks
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        self.escalation_task = asyncio.create_task(self._escalation_loop())
        
        self.logger.info("Alert manager started", operation="start_alert_manager")
    
    async def stop(self):
        """Stop the alert manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.evaluation_task:
            self.evaluation_task.cancel()
        if self.escalation_task:
            self.escalation_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.evaluation_task, self.escalation_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop notification sender
        await self.notification_sender.stop()
        
        self.logger.info("Alert manager stopped", operation="stop_alert_manager")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.id] = rule
        self.rule_states[rule.id] = {
            'condition_start_time': None,
            'last_evaluation': None,
            'consecutive_violations': 0
        }
        
        self.logger.info(
            f"Added alert rule: {rule.name}",
            operation="add_alert_rule",
            rule_id=rule.id,
            metric=rule.metric_name
        )
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            rule = self.alert_rules.pop(rule_id)
            self.rule_states.pop(rule_id, None)
            
            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]
            
            for alert in alerts_to_resolve:
                alert.resolve()
                self.active_alerts.pop(alert.id, None)
                self.alert_history.append(alert)
            
            self.logger.info(
                f"Removed alert rule: {rule.name}",
                operation="remove_alert_rule",
                rule_id=rule_id
            )
    
    def add_notification_channel(self, name: str, config: NotificationChannelConfig):
        """Add a notification channel."""
        self.notification_channels[name] = config
        
        self.logger.info(
            f"Added notification channel: {name}",
            operation="add_notification_channel",
            channel_type=config.channel.value
        )
    
    async def _evaluation_loop(self):
        """Background task to evaluate alert rules."""
        while self.running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in evaluation loop: {e}",
                    operation="evaluation_loop"
                )
                await asyncio.sleep(30)
    
    async def _escalation_loop(self):
        """Background task to handle alert escalations."""
        while self.running:
            try:
                await self._process_escalations()
                await asyncio.sleep(60)  # Check escalations every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in escalation loop: {e}",
                    operation="escalation_loop"
                )
                await asyncio.sleep(60)
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules."""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
                self.stats['rules_evaluated'] += 1
            except Exception as e:
                self.logger.error(
                    f"Error evaluating rule {rule_id}: {e}",
                    operation="evaluate_rule",
                    rule_id=rule_id
                )
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get metric series
        metric_series = self.metrics_collector.get_metric_series(rule.metric_name, rule.tags)
        
        if not metric_series:
            return
        
        # Get latest value
        latest_value = metric_series.get_latest_value()
        if not latest_value:
            return
        
        # Check if value is recent enough
        age_seconds = (datetime.utcnow() - latest_value.timestamp).total_seconds()
        if age_seconds > rule.evaluation_interval * 2:  # Allow some staleness
            return
        
        # Evaluate condition
        condition_met = rule.evaluate(latest_value.value)
        rule_state = self.rule_states[rule.id]
        now = datetime.utcnow()
        
        if condition_met:
            # Condition is met
            if rule_state['condition_start_time'] is None:
                rule_state['condition_start_time'] = now
                rule_state['consecutive_violations'] = 1
            else:
                rule_state['consecutive_violations'] += 1
            
            # Check if duration threshold is met
            duration_minutes = (now - rule_state['condition_start_time']).total_seconds() / 60
            
            if duration_minutes >= rule.duration_minutes:
                # Create or update alert
                await self._create_or_update_alert(rule, latest_value.value)
        else:
            # Condition is not met
            if rule_state['condition_start_time'] is not None:
                # Resolve any active alert
                await self._resolve_alert(rule.id)
            
            # Reset state
            rule_state['condition_start_time'] = None
            rule_state['consecutive_violations'] = 0
        
        rule_state['last_evaluation'] = now
    
    async def _create_or_update_alert(self, rule: AlertRule, current_value: Union[int, float]):
        """Create or update an alert."""
        # Check if alert already exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.id and alert.is_active():
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.updated_at = datetime.utcnow()
        else:
            # Create new alert
            alert_id = f"{rule.id}_{int(datetime.utcnow().timestamp())}"
            
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                rule_name=rule.name,
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold=rule.threshold,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=rule.tags,
                labels=rule.labels
            )
            
            self.active_alerts[alert_id] = alert
            self.stats['alerts_created'] += 1
            
            # Send notifications
            await self._send_alert_notifications(alert, rule)
            
            self.logger.warning(
                f"Alert created: {rule.name}",
                operation="create_alert",
                alert_id=alert_id,
                rule_id=rule.id,
                current_value=current_value,
                threshold=rule.threshold
            )
    
    async def _resolve_alert(self, rule_id: str):
        """Resolve alerts for a rule."""
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values()
            if alert.rule_id == rule_id and alert.is_active()
        ]
        
        for alert in alerts_to_resolve:
            alert.resolve()
            self.active_alerts.pop(alert.id, None)
            self.alert_history.append(alert)
            self.stats['alerts_resolved'] += 1
            
            self.logger.info(
                f"Alert resolved: {alert.rule_name}",
                operation="resolve_alert",
                alert_id=alert.id,
                duration=str(alert.get_duration())
            )
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert."""
        for channel_type in rule.notification_channels:
            # Find matching notification channel config
            channel_config = None
            for name, config in self.notification_channels.items():
                if config.channel == channel_type:
                    channel_config = config
                    break
            
            if channel_config:
                success = await self.notification_sender.send_notification(alert, channel_config)
                if success:
                    self.stats['notifications_sent'] += 1
    
    async def _process_escalations(self):
        """Process alert escalations."""
        for alert in self.active_alerts.values():
            if not alert.is_active():
                continue
            
            rule = self.alert_rules.get(alert.rule_id)
            if not rule or not rule.escalation_policy:
                continue
            
            escalation_policy = self.escalation_policies.get(rule.escalation_policy)
            if not escalation_policy:
                continue
            
            # Check if escalation is needed
            duration_minutes = alert.get_duration().total_seconds() / 60
            new_level = escalation_policy.get_escalation_level(duration_minutes)
            
            if new_level > alert.escalation_level:
                alert.escalation_level = new_level
                
                # Send escalation notifications
                if new_level < len(escalation_policy.levels):
                    level_config = escalation_policy.levels[new_level]
                    # Process escalation notifications
                    # (Implementation would depend on specific escalation requirements)
                
                self.logger.warning(
                    f"Alert escalated to level {new_level}",
                    operation="escalate_alert",
                    alert_id=alert.id,
                    escalation_level=new_level
                )
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge(acknowledged_by)
            
            self.logger.info(
                f"Alert acknowledged: {alert.rule_name}",
                operation="acknowledge_alert",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by
            )
            return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts."""
        alerts = [alert for alert in self.active_alerts.values() if alert.is_active()]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return list(self.alert_history)[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        active_alerts_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            if alert.is_active():
                active_alerts_by_severity[alert.severity.value] += 1
        
        return {
            **self.stats,
            'active_alerts_total': len([a for a in self.active_alerts.values() if a.is_active()]),
            'active_alerts_by_severity': dict(active_alerts_by_severity),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'notification_channels': len(self.notification_channels),
            'uptime_seconds': (
                (datetime.utcnow() - self.stats['start_time']).total_seconds()
                if self.stats['start_time'] else 0
            )
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# Convenience functions
async def create_alert(
    rule_name: str,
    metric_name: str,
    condition: str,
    threshold: Union[int, float],
    severity: AlertSeverity,
    description: str = "",
    duration_minutes: int = 5
) -> str:
    """Create a simple alert rule."""
    alert_manager = get_alert_manager()
    
    rule_id = f"custom_{rule_name.lower().replace(' ', '_')}"
    rule = AlertRule(
        id=rule_id,
        name=rule_name,
        description=description or f"{rule_name} alert",
        metric_name=metric_name,
        condition=condition,
        threshold=threshold,
        severity=severity,
        duration_minutes=duration_minutes
    )
    
    alert_manager.add_alert_rule(rule)
    return rule_id


async def setup_email_notifications(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    from_email: str,
    to_emails: List[str]
):
    """Setup email notifications."""
    alert_manager = get_alert_manager()
    
    config = NotificationChannelConfig(
        channel=NotificationChannel.EMAIL,
        config={
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_email': from_email,
            'to_emails': to_emails
        }
    )
    
    alert_manager.add_notification_channel('email', config)


async def setup_slack_notifications(webhook_url: str):
    """Setup Slack notifications."""
    alert_manager = get_alert_manager()
    
    config = NotificationChannelConfig(
        channel=NotificationChannel.SLACK,
        config={'webhook_url': webhook_url}
    )
    
    alert_manager.add_notification_channel('slack', config)