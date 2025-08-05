#!/usr/bin/env python3
"""
Project Synapse Learning Scrapers - GitHub Actions Monitor
Advanced monitoring and alerting for learning scraper workflows.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import click

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowRun:
    """Represents a workflow run."""
    id: int
    status: str
    conclusion: Optional[str]
    created_at: datetime
    updated_at: datetime
    html_url: str
    head_commit_message: str
    run_number: int
    
    @property
    def duration(self) -> timedelta:
        """Get run duration."""
        return self.updated_at - self.created_at
    
    @property
    def is_completed(self) -> bool:
        """Check if run is completed."""
        return self.status == 'completed'
    
    @property
    def is_successful(self) -> bool:
        """Check if run was successful."""
        return self.conclusion == 'success'


@dataclass
class MonitoringAlert:
    """Represents a monitoring alert."""
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: datetime
    workflow_run: Optional[WorkflowRun] = None
    metadata: Dict[str, Any] = None


class GitHubActionsMonitor:
    """
    Advanced monitor for GitHub Actions workflows.
    
    Provides real-time monitoring, alerting, and performance tracking
    for learning scraper workflows.
    """
    
    def __init__(
        self,
        github_token: str,
        repository: str,
        workflow_name: str = "learning-scrapers.yml",
        check_interval: int = 60,
        alert_webhook: Optional[str] = None
    ):
        """
        Initialize monitor.
        
        Args:
            github_token: GitHub personal access token
            repository: Repository in format owner/repo
            workflow_name: Workflow filename
            check_interval: Check interval in seconds
            alert_webhook: Webhook URL for alerts
        """
        self.github_token = github_token
        self.repository = repository
        self.workflow_name = workflow_name
        self.check_interval = check_interval
        self.alert_webhook = alert_webhook
        
        # Monitoring state
        self.last_check: Optional[datetime] = None
        self.known_runs: Dict[int, WorkflowRun] = {}
        self.alerts: List[MonitoringAlert] = []
        self.stats: Dict[str, Any] = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_duration': 0,
            'last_success': None,
            'last_failure': None
        }
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start the monitor."""
        logger.info("Starting GitHub Actions monitor...")
        
        # Create HTTP session
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Project-Synapse-Monitor/1.0'
        }
        
        self.session = aiohttp.ClientSession(headers=headers)
        
        # Initial data load
        await self._load_initial_data()
        
        logger.info(f"Monitor started for {self.repository}")
    
    async def stop(self):
        """Stop the monitor."""
        if self.session:
            await self.session.close()
        
        logger.info("Monitor stopped")
    
    async def monitor_continuous(self):
        """Run continuous monitoring."""
        logger.info(f"Starting continuous monitoring (interval: {self.check_interval}s)")
        
        try:
            while True:
                await self._check_workflows()
                await asyncio.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            raise
    
    async def check_once(self) -> Dict[str, Any]:
        """Perform a single check and return results."""
        await self._check_workflows()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'stats': self.stats.copy(),
            'recent_runs': [
                {
                    'id': run.id,
                    'status': run.status,
                    'conclusion': run.conclusion,
                    'created_at': run.created_at.isoformat(),
                    'duration_seconds': run.duration.total_seconds(),
                    'url': run.html_url
                }
                for run in list(self.known_runs.values())[-5:]
            ],
            'recent_alerts': [
                {
                    'level': alert.level,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.alerts[-10:]
            ]
        }
    
    async def get_workflow_logs(self, run_id: int) -> Optional[bytes]:
        """Download workflow logs."""
        try:
            url = f"https://api.github.com/repos/{self.repository}/actions/runs/{run_id}/logs"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to get logs for run {run_id}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting logs for run {run_id}: {e}")
            return None
    
    async def get_workflow_artifacts(self, run_id: int) -> List[Dict[str, Any]]:
        """Get workflow artifacts."""
        try:
            url = f"https://api.github.com/repos/{self.repository}/actions/runs/{run_id}/artifacts"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('artifacts', [])
                else:
                    logger.error(f"Failed to get artifacts for run {run_id}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting artifacts for run {run_id}: {e}")
            return []
    
    async def trigger_workflow(
        self,
        inputs: Dict[str, Any],
        ref: str = "main"
    ) -> bool:
        """Trigger a workflow run."""
        try:
            url = f"https://api.github.com/repos/{self.repository}/actions/workflows/{self.workflow_name}/dispatches"
            
            payload = {
                'ref': ref,
                'inputs': inputs
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 204:
                    logger.info("Workflow triggered successfully")
                    return True
                else:
                    logger.error(f"Failed to trigger workflow: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error triggering workflow: {e}")
            return False
    
    async def _load_initial_data(self):
        """Load initial workflow data."""
        try:
            runs = await self._fetch_workflow_runs(per_page=50)
            
            for run_data in runs:
                run = self._parse_workflow_run(run_data)
                self.known_runs[run.id] = run
            
            self._update_stats()
            
            logger.info(f"Loaded {len(self.known_runs)} workflow runs")
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
    
    async def _check_workflows(self):
        """Check for workflow updates."""
        try:
            # Get recent runs
            runs = await self._fetch_workflow_runs(per_page=20)
            
            new_runs = []
            updated_runs = []
            
            for run_data in runs:
                run = self._parse_workflow_run(run_data)
                
                if run.id not in self.known_runs:
                    # New run
                    new_runs.append(run)
                    self.known_runs[run.id] = run
                elif self.known_runs[run.id].status != run.status:
                    # Updated run
                    updated_runs.append(run)
                    self.known_runs[run.id] = run
            
            # Process new and updated runs
            for run in new_runs:
                await self._handle_new_run(run)
            
            for run in updated_runs:
                await self._handle_updated_run(run)
            
            # Update statistics
            self._update_stats()
            
            self.last_check = datetime.utcnow()
            
            if new_runs or updated_runs:
                logger.info(f"Check completed: {len(new_runs)} new, {len(updated_runs)} updated")
            
        except Exception as e:
            logger.error(f"Error checking workflows: {e}")
            await self._create_alert('error', f"Monitoring check failed: {e}")
    
    async def _fetch_workflow_runs(self, per_page: int = 20) -> List[Dict[str, Any]]:
        """Fetch workflow runs from GitHub API."""
        url = f"https://api.github.com/repos/{self.repository}/actions/workflows/{self.workflow_name}/runs"
        params = {'per_page': per_page}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('workflow_runs', [])
            else:
                raise Exception(f"API request failed: {response.status}")
    
    def _parse_workflow_run(self, run_data: Dict[str, Any]) -> WorkflowRun:
        """Parse workflow run data."""
        return WorkflowRun(
            id=run_data['id'],
            status=run_data['status'],
            conclusion=run_data.get('conclusion'),
            created_at=datetime.fromisoformat(run_data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(run_data['updated_at'].replace('Z', '+00:00')),
            html_url=run_data['html_url'],
            head_commit_message=run_data.get('head_commit', {}).get('message', ''),
            run_number=run_data['run_number']
        )
    
    async def _handle_new_run(self, run: WorkflowRun):
        """Handle a new workflow run."""
        logger.info(f"New workflow run: #{run.run_number} (ID: {run.id})")
        
        await self._create_alert(
            'info',
            f"New workflow run started: #{run.run_number}",
            workflow_run=run
        )
    
    async def _handle_updated_run(self, run: WorkflowRun):
        """Handle an updated workflow run."""
        logger.info(f"Workflow run updated: #{run.run_number} - {run.status}")
        
        if run.is_completed:
            if run.is_successful:
                await self._create_alert(
                    'info',
                    f"Workflow run completed successfully: #{run.run_number}",
                    workflow_run=run
                )
            else:
                await self._create_alert(
                    'error',
                    f"Workflow run failed: #{run.run_number} - {run.conclusion}",
                    workflow_run=run
                )
    
    def _update_stats(self):
        """Update monitoring statistics."""
        if not self.known_runs:
            return
        
        completed_runs = [run for run in self.known_runs.values() if run.is_completed]
        successful_runs = [run for run in completed_runs if run.is_successful]
        failed_runs = [run for run in completed_runs if not run.is_successful]
        
        self.stats.update({
            'total_runs': len(self.known_runs),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'completion_rate': len(completed_runs) / len(self.known_runs) if self.known_runs else 0,
            'success_rate': len(successful_runs) / len(completed_runs) if completed_runs else 0
        })
        
        # Calculate average duration
        if completed_runs:
            avg_duration = sum(run.duration.total_seconds() for run in completed_runs) / len(completed_runs)
            self.stats['avg_duration_seconds'] = avg_duration
        
        # Update last success/failure
        if successful_runs:
            self.stats['last_success'] = max(successful_runs, key=lambda r: r.updated_at).updated_at.isoformat()
        
        if failed_runs:
            self.stats['last_failure'] = max(failed_runs, key=lambda r: r.updated_at).updated_at.isoformat()
    
    async def _create_alert(
        self,
        level: str,
        message: str,
        workflow_run: Optional[WorkflowRun] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a monitoring alert."""
        alert = MonitoringAlert(
            level=level,
            message=message,
            timestamp=datetime.utcnow(),
            workflow_run=workflow_run,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Send webhook notification if configured
        if self.alert_webhook and level in ['error', 'critical']:
            await self._send_webhook_alert(alert)
        
        # Log alert
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"Alert: {message}")
    
    async def _send_webhook_alert(self, alert: MonitoringAlert):
        """Send alert to webhook."""
        try:
            payload = {
                'level': alert.level,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'repository': self.repository,
                'workflow': self.workflow_name
            }
            
            if alert.workflow_run:
                payload['workflow_run'] = {
                    'id': alert.workflow_run.id,
                    'number': alert.workflow_run.run_number,
                    'status': alert.workflow_run.status,
                    'conclusion': alert.workflow_run.conclusion,
                    'url': alert.workflow_run.html_url
                }
            
            async with self.session.post(self.alert_webhook, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Webhook alert failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")


@click.group()
def cli():
    """Project Synapse Learning Scrapers - GitHub Actions Monitor"""
    pass


@cli.command()
@click.option('--token', envvar='GITHUB_TOKEN', required=True, help='GitHub token')
@click.option('--repo', envvar='GITHUB_REPOSITORY', required=True, help='Repository (owner/repo)')
@click.option('--workflow', default='learning-scrapers.yml', help='Workflow filename')
@click.option('--interval', default=60, help='Check interval in seconds')
@click.option('--webhook', help='Alert webhook URL')
def monitor(token, repo, workflow, interval, webhook):
    """Start continuous monitoring."""
    async def run_monitor():
        async with GitHubActionsMonitor(
            github_token=token,
            repository=repo,
            workflow_name=workflow,
            check_interval=interval,
            alert_webhook=webhook
        ) as monitor:
            await monitor.monitor_continuous()
    
    asyncio.run(run_monitor())


@cli.command()
@click.option('--token', envvar='GITHUB_TOKEN', required=True, help='GitHub token')
@click.option('--repo', envvar='GITHUB_REPOSITORY', required=True, help='Repository (owner/repo)')
@click.option('--workflow', default='learning-scrapers.yml', help='Workflow filename')
@click.option('--output', help='Output file for results')
def check(token, repo, workflow, output):
    """Perform a single check."""
    async def run_check():
        async with GitHubActionsMonitor(
            github_token=token,
            repository=repo,
            workflow_name=workflow
        ) as monitor:
            results = await monitor.check_once()
            
            if output:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output}")
            else:
                print(json.dumps(results, indent=2))
    
    asyncio.run(run_check())


@cli.command()
@click.option('--token', envvar='GITHUB_TOKEN', required=True, help='GitHub token')
@click.option('--repo', envvar='GITHUB_REPOSITORY', required=True, help='Repository (owner/repo)')
@click.option('--workflow', default='learning-scrapers.yml', help='Workflow filename')
@click.option('--urls', required=True, help='Comma-separated URLs to scrape')
@click.option('--scraper-type', default='playwright', help='Scraper type')
@click.option('--learning/--no-learning', default=True, help='Enable learning mode')
@click.option('--proxy/--no-proxy', default=False, help='Use proxy network')
def trigger(token, repo, workflow, urls, scraper_type, learning, proxy):
    """Trigger a workflow run."""
    async def run_trigger():
        async with GitHubActionsMonitor(
            github_token=token,
            repository=repo,
            workflow_name=workflow
        ) as monitor:
            inputs = {
                'target_urls': urls,
                'scraper_type': scraper_type,
                'learning_mode': learning,
                'use_proxy': proxy
            }
            
            success = await monitor.trigger_workflow(inputs)
            
            if success:
                print("Workflow triggered successfully")
            else:
                print("Failed to trigger workflow")
                exit(1)
    
    asyncio.run(run_trigger())


if __name__ == '__main__':
    cli()