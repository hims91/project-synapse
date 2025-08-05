"""
Cache warming system for Project Synapse.

Proactively loads frequently accessed data into cache layers
to improve performance and reduce database load.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json

from ..logging_config import get_logger
from ..metrics_collector import get_metrics_collector
from .cache_manager import get_cache_manager, MultiLayerCacheManager


class WarmingStrategy(str, Enum):
    """Cache warming strategy enumeration."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"
    PREDICTIVE = "predictive"


@dataclass
class WarmingJob:
    """Cache warming job configuration."""
    name: str
    namespace: str
    strategy: WarmingStrategy
    data_loader: Callable[[], Awaitable[Dict[str, Any]]]
    ttl: Optional[int] = None
    schedule_interval: Optional[int] = None  # seconds
    priority: int = 1  # 1 = highest, 10 = lowest
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_duration: float = 0.0
    tags: List[str] = field(default_factory=list)


class CacheWarmer:
    """Cache warming system for proactive data loading."""
    
    def __init__(self, cache_manager: Optional[MultiLayerCacheManager] = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self.logger = get_logger(__name__, 'cache_warmer')
        self.metrics = get_metrics_collector()
        
        self.jobs: Dict[str, WarmingJob] = {}
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        self.stats = {
            'jobs_registered': 0,
            'jobs_executed': 0,
            'jobs_succeeded': 0,
            'jobs_failed': 0,
            'total_items_warmed': 0,
            'total_warming_time': 0.0
        }
    
    def register_job(self, job: WarmingJob) -> None:
        """Register a cache warming job."""
        self.jobs[job.name] = job
        self.stats['jobs_registered'] += 1
        
        # Schedule next run for scheduled jobs
        if job.strategy == WarmingStrategy.SCHEDULED and job.schedule_interval:
            job.next_run = datetime.utcnow() + timedelta(seconds=job.schedule_interval)
        
        self.logger.info(f"Registered cache warming job: {job.name}")
    
    def unregister_job(self, job_name: str) -> bool:
        """Unregister a cache warming job."""
        if job_name in self.jobs:
            del self.jobs[job_name]
            self.logger.info(f"Unregistered cache warming job: {job_name}")
            return True
        return False
    
    def get_job(self, job_name: str) -> Optional[WarmingJob]:
        """Get a warming job by name."""
        return self.jobs.get(job_name)
    
    def list_jobs(self, strategy: Optional[WarmingStrategy] = None, enabled_only: bool = True) -> List[WarmingJob]:
        """List warming jobs with optional filtering."""
        jobs = list(self.jobs.values())
        
        if strategy:
            jobs = [job for job in jobs if job.strategy == strategy]
        
        if enabled_only:
            jobs = [job for job in jobs if job.enabled]
        
        return sorted(jobs, key=lambda j: j.priority)
    
    async def warm_job(self, job_name: str) -> bool:
        """Execute a specific warming job."""
        job = self.jobs.get(job_name)
        if not job or not job.enabled:
            self.logger.warning(f"Job {job_name} not found or disabled")
            return False
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting cache warming job: {job_name}")
            
            # Load data using the job's data loader
            data = await job.data_loader()
            
            if not data:
                self.logger.warning(f"No data returned from job {job_name}")
                return True
            
            # Warm cache with the loaded data
            success = await self.cache_manager.set_multiple(
                namespace=job.namespace,
                data=data,
                ttl=job.ttl
            )
            
            # Update job statistics
            duration = time.time() - start_time
            job.run_count += 1
            job.last_run = datetime.utcnow()
            
            if success:
                job.success_count += 1
                self.stats['jobs_succeeded'] += 1
                self.stats['total_items_warmed'] += len(data)
                
                self.logger.info(
                    f"Cache warming job {job_name} completed successfully: "
                    f"{len(data)} items in {duration:.2f}s"
                )
            else:
                job.error_count += 1
                self.stats['jobs_failed'] += 1
                self.logger.error(f"Cache warming job {job_name} failed to set cache data")
            
            # Update average duration
            job.avg_duration = (job.avg_duration * (job.run_count - 1) + duration) / job.run_count
            
            # Schedule next run for scheduled jobs
            if job.strategy == WarmingStrategy.SCHEDULED and job.schedule_interval:
                job.next_run = datetime.utcnow() + timedelta(seconds=job.schedule_interval)
            
            self.stats['jobs_executed'] += 1
            self.stats['total_warming_time'] += duration
            
            # Record metrics
            counter = self.metrics.get_counter('cache_warming_jobs_total')
            counter.increment(1, job=job_name, status='success' if success else 'failed')
            
            gauge = self.metrics.get_gauge('cache_warming_duration_seconds')
            gauge.set(duration, job=job_name)
            
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            job.run_count += 1
            job.error_count += 1
            job.last_run = datetime.utcnow()
            
            self.stats['jobs_failed'] += 1
            self.stats['jobs_executed'] += 1
            
            self.logger.error(f"Cache warming job {job_name} failed: {e}")
            
            # Record error metrics
            counter = self.metrics.get_counter('cache_warming_jobs_total')
            counter.increment(1, job=job_name, status='error')
            
            return False
    
    async def warm_namespace(self, namespace: str) -> int:
        """Warm all jobs for a specific namespace."""
        jobs = [job for job in self.jobs.values() if job.namespace == namespace and job.enabled]
        
        if not jobs:
            self.logger.info(f"No warming jobs found for namespace: {namespace}")
            return 0
        
        self.logger.info(f"Warming {len(jobs)} jobs for namespace: {namespace}")
        
        # Execute jobs in priority order
        jobs.sort(key=lambda j: j.priority)
        success_count = 0
        
        for job in jobs:
            if await self.warm_job(job.name):
                success_count += 1
        
        self.logger.info(f"Completed warming for namespace {namespace}: {success_count}/{len(jobs)} successful")
        return success_count
    
    async def warm_all_immediate(self) -> int:
        """Warm all immediate strategy jobs."""
        jobs = [job for job in self.jobs.values() 
                if job.strategy == WarmingStrategy.IMMEDIATE and job.enabled]
        
        if not jobs:
            return 0
        
        self.logger.info(f"Warming {len(jobs)} immediate jobs")
        
        # Execute jobs concurrently for immediate strategy
        tasks = [self.warm_job(job.name) for job in jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        self.logger.info(f"Completed immediate warming: {success_count}/{len(jobs)} successful")
        
        return success_count
    
    async def start_scheduler(self) -> None:
        """Start the cache warming scheduler."""
        if self.running:
            self.logger.warning("Cache warming scheduler is already running")
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._scheduler_worker())
        self.logger.info("Cache warming scheduler started")
    
    async def stop_scheduler(self) -> None:
        """Stop the cache warming scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
        
        self.logger.info("Cache warming scheduler stopped")
    
    async def _scheduler_worker(self) -> None:
        """Background worker for scheduled cache warming."""
        self.logger.info("Cache warming scheduler worker started")
        
        while self.running:
            try:
                now = datetime.utcnow()
                
                # Find jobs that need to run
                jobs_to_run = [
                    job for job in self.jobs.values()
                    if (job.strategy == WarmingStrategy.SCHEDULED and 
                        job.enabled and 
                        job.next_run and 
                        now >= job.next_run)
                ]
                
                if jobs_to_run:
                    self.logger.info(f"Running {len(jobs_to_run)} scheduled warming jobs")
                    
                    # Sort by priority
                    jobs_to_run.sort(key=lambda j: j.priority)
                    
                    # Execute jobs
                    for job in jobs_to_run:
                        await self.warm_job(job.name)
                
                # Sleep for a short interval before checking again
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache warming scheduler: {e}")
                await asyncio.sleep(60)  # Wait longer on error
        
        self.logger.info("Cache warming scheduler worker stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics."""
        job_stats = {}
        for name, job in self.jobs.items():
            job_stats[name] = {
                'strategy': job.strategy.value,
                'namespace': job.namespace,
                'enabled': job.enabled,
                'run_count': job.run_count,
                'success_count': job.success_count,
                'error_count': job.error_count,
                'success_rate': job.success_count / job.run_count if job.run_count > 0 else 0,
                'avg_duration': job.avg_duration,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'next_run': job.next_run.isoformat() if job.next_run else None
            }
        
        return {
            'overall': self.stats,
            'jobs': job_stats,
            'scheduler_running': self.running
        }


# Predefined warming jobs for common data
async def load_popular_articles() -> Dict[str, Any]:
    """Load popular articles for cache warming."""
    # This would typically query the database for popular articles
    # For now, we'll return a placeholder
    return {
        'popular_articles': [
            {'id': 1, 'title': 'Popular Article 1', 'views': 1000},
            {'id': 2, 'title': 'Popular Article 2', 'views': 800},
            {'id': 3, 'title': 'Popular Article 3', 'views': 600},
        ]
    }


async def load_trending_topics() -> Dict[str, Any]:
    """Load trending topics for cache warming."""
    return {
        'trending_topics': [
            {'topic': 'AI', 'score': 95},
            {'topic': 'Technology', 'score': 87},
            {'topic': 'Science', 'score': 76},
        ]
    }


async def load_user_preferences() -> Dict[str, Any]:
    """Load common user preferences for cache warming."""
    return {
        'default_preferences': {
            'theme': 'light',
            'language': 'en',
            'notifications': True
        }
    }


def create_default_warming_jobs() -> List[WarmingJob]:
    """Create default cache warming jobs."""
    return [
        WarmingJob(
            name="popular_articles",
            namespace="articles",
            strategy=WarmingStrategy.SCHEDULED,
            data_loader=load_popular_articles,
            ttl=3600,  # 1 hour
            schedule_interval=1800,  # 30 minutes
            priority=1,
            tags=["articles", "popular"]
        ),
        WarmingJob(
            name="trending_topics",
            namespace="trends",
            strategy=WarmingStrategy.SCHEDULED,
            data_loader=load_trending_topics,
            ttl=1800,  # 30 minutes
            schedule_interval=900,  # 15 minutes
            priority=2,
            tags=["trends", "topics"]
        ),
        WarmingJob(
            name="user_preferences",
            namespace="users",
            strategy=WarmingStrategy.IMMEDIATE,
            data_loader=load_user_preferences,
            ttl=7200,  # 2 hours
            priority=3,
            tags=["users", "preferences"]
        )
    ]


# Global cache warmer instance
_cache_warmer: Optional[CacheWarmer] = None


def get_cache_warmer() -> CacheWarmer:
    """Get the global cache warmer instance."""
    global _cache_warmer
    if _cache_warmer is None:
        _cache_warmer = CacheWarmer()
        
        # Register default warming jobs
        for job in create_default_warming_jobs():
            _cache_warmer.register_job(job)
    
    return _cache_warmer