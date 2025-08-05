#!/usr/bin/env python3
"""
Health Check Script for Project Synapse Scrapers Container
Performs comprehensive health checks for containerized scrapers.
"""
import asyncio
import sys
import os
import time
import json
from typing import Dict, Any
from urllib.parse import urlparse

# Add src to path for imports
sys.path.insert(0, '/app')

try:
    from src.synaptic_vesicle.database import DatabaseManager
    from src.neurons.http_scraper import HTTPScraper
    from src.neurons.recipe_engine import ScrapingRecipeEngine
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class HealthChecker:
    """Comprehensive health checker for scraper containers."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checks = {}
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            db_manager = DatabaseManager()
            health = await db_manager.health_check()
            await db_manager.close()
            
            return {
                'status': 'healthy' if health['status'] == 'healthy' else 'unhealthy',
                'details': health,
                'check_time': time.time() - self.start_time
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'check_time': time.time() - self.start_time
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            import redis
            
            redis_url = os.getenv('REDIS_URL')
            if not redis_url:
                return {
                    'status': 'skipped',
                    'reason': 'REDIS_URL not configured',
                    'check_time': time.time() - self.start_time
                }
            
            parsed = urlparse(redis_url)
            r = redis.Redis(
                host=parsed.hostname,
                port=parsed.port or 6379,
                password=parsed.password,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            r.ping()
            
            return {
                'status': 'healthy',
                'host': parsed.hostname,
                'port': parsed.port or 6379,
                'check_time': time.time() - self.start_time
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'check_time': time.time() - self.start_time
            }
    
    async def check_scraper_components(self) -> Dict[str, Any]:
        """Check scraper components initialization."""
        try:
            # Test HTTP scraper
            scraper = HTTPScraper(timeout=5.0)
            
            # Test recipe engine
            recipe_engine = ScrapingRecipeEngine()
            
            # Basic functionality test
            test_html = "<html><head><title>Test</title></head><body><h1>Test</h1><p>Content</p></body></html>"
            
            # Test recipe engine validation
            from src.shared.schemas import ScrapingSelectors, ScrapingAction
            test_selectors = ScrapingSelectors(title="h1", content="p")
            is_valid, errors = recipe_engine.validate_recipe(test_selectors, [])
            
            await scraper.close()
            
            return {
                'status': 'healthy',
                'components': {
                    'http_scraper': {
                        'timeout': scraper.timeout,
                        'max_retries': scraper.max_retries,
                        'user_agents': len(scraper.USER_AGENTS)
                    },
                    'recipe_engine': {
                        'cache_ttl': str(recipe_engine.cache_ttl),
                        'validation_working': is_valid
                    }
                },
                'check_time': time.time() - self.start_time
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'check_time': time.time() - self.start_time
            }
    
    async def check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem health and permissions."""
        try:
            # Check required directories
            required_dirs = ['/app/logs', '/app/data', '/app/tmp']
            dir_status = {}
            
            for dir_path in required_dirs:
                try:
                    # Check if directory exists and is writable
                    if os.path.exists(dir_path):
                        if os.access(dir_path, os.W_OK):
                            dir_status[dir_path] = 'writable'
                        else:
                            dir_status[dir_path] = 'read-only'
                    else:
                        dir_status[dir_path] = 'missing'
                except Exception as e:
                    dir_status[dir_path] = f'error: {e}'
            
            # Check disk space
            statvfs = os.statvfs('/app')
            free_space = statvfs.f_frsize * statvfs.f_bavail
            total_space = statvfs.f_frsize * statvfs.f_blocks
            used_percentage = ((total_space - free_space) / total_space) * 100
            
            # Determine overall status
            all_dirs_ok = all(status in ['writable', 'read-only'] for status in dir_status.values())
            disk_ok = used_percentage < 90  # Alert if disk usage > 90%
            
            status = 'healthy' if all_dirs_ok and disk_ok else 'unhealthy'
            
            return {
                'status': status,
                'directories': dir_status,
                'disk_usage': {
                    'used_percentage': round(used_percentage, 2),
                    'free_bytes': free_space,
                    'total_bytes': total_space
                },
                'check_time': time.time() - self.start_time
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'check_time': time.time() - self.start_time
            }
    
    async def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration."""
        try:
            required_vars = ['DATABASE_URL', 'REDIS_URL']
            optional_vars = [
                'SCRAPER_WORKERS', 'SCRAPER_TIMEOUT', 'SCRAPER_MAX_RETRIES',
                'LOG_LEVEL', 'HEALTH_CHECK_PORT'
            ]
            
            env_status = {}
            missing_required = []
            
            # Check required variables
            for var in required_vars:
                if os.getenv(var):
                    env_status[var] = 'configured'
                else:
                    env_status[var] = 'missing'
                    missing_required.append(var)
            
            # Check optional variables
            for var in optional_vars:
                env_status[var] = 'configured' if os.getenv(var) else 'default'
            
            status = 'healthy' if not missing_required else 'unhealthy'
            
            result = {
                'status': status,
                'environment_variables': env_status,
                'check_time': time.time() - self.start_time
            }
            
            if missing_required:
                result['missing_required'] = missing_required
            
            return result
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'check_time': time.time() - self.start_time
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        print("Running health checks...")
        
        # Run checks concurrently
        checks = await asyncio.gather(
            self.check_environment(),
            self.check_filesystem(),
            self.check_database(),
            self.check_redis(),
            self.check_scraper_components(),
            return_exceptions=True
        )
        
        check_names = [
            'environment',
            'filesystem', 
            'database',
            'redis',
            'scraper_components'
        ]
        
        results = {}
        overall_healthy = True
        
        for name, check_result in zip(check_names, checks):
            if isinstance(check_result, Exception):
                results[name] = {
                    'status': 'error',
                    'error': str(check_result),
                    'check_time': time.time() - self.start_time
                }
                overall_healthy = False
            else:
                results[name] = check_result
                if check_result.get('status') not in ['healthy', 'skipped']:
                    overall_healthy = False
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'checks': results,
            'version': os.getenv('VERSION', 'unknown'),
            'build_date': os.getenv('BUILD_DATE', 'unknown')
        }


async def main():
    """Main health check execution."""
    checker = HealthChecker()
    
    try:
        # Run health checks
        results = await checker.run_all_checks()
        
        # Output results
        if os.getenv('HEALTH_CHECK_FORMAT', 'json').lower() == 'json':
            print(json.dumps(results, indent=2))
        else:
            # Human-readable format
            print(f"Overall Status: {results['overall_status'].upper()}")
            print(f"Uptime: {results['uptime_seconds']:.2f} seconds")
            print("\nComponent Status:")
            
            for component, check in results['checks'].items():
                status = check.get('status', 'unknown').upper()
                print(f"  {component}: {status}")
                
                if check.get('error'):
                    print(f"    Error: {check['error']}")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_status'] == 'healthy' else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())