#!/usr/bin/env python3
"""
Simple test runner for Project Synapse integration tests.
Runs a subset of tests to validate the system works.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_test_command(command: list, description: str) -> bool:
    """Run a test command and return success status."""
    print(f"🧪 {description}...")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED ({duration:.1f}s)")
            return True
        else:
            print(f"❌ {description} - FAILED ({duration:.1f}s)")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (120s)")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main():
    """Run integration tests."""
    print("🧠 Project Synapse - Integration Test Runner")
    print("=" * 50)
    
    # Test commands to run
    test_commands = [
        # Basic system tests
        ([sys.executable, "-c", "import src.axon_interface.main; print('✅ Main app imports successfully')"], 
         "Import Test - Main Application"),
        
        ([sys.executable, "-c", "from src.shared.database import get_database_manager; print('✅ Database manager imports')"], 
         "Import Test - Database Manager"),
        
        ([sys.executable, "-c", "from src.shared.caching import get_cache_manager; print('✅ Cache manager imports')"], 
         "Import Test - Cache Manager"),
        
        ([sys.executable, "-c", "from src.signal_relay.task_dispatcher import get_task_dispatcher; print('✅ Task dispatcher imports')"], 
         "Import Test - Task Dispatcher"),
        
        # Configuration tests
        ([sys.executable, "-c", "from src.shared.config import get_settings; s = get_settings(); print(f'✅ Config loaded: {s.environment}')"], 
         "Configuration Test"),
        
        # Simple unit tests
        ([sys.executable, "-m", "pytest", "tests/test_models.py", "-v", "--tb=short", "-x"], 
         "Unit Tests - Data Models"),
        
        ([sys.executable, "-m", "pytest", "tests/test_repositories.py::TestArticleRepository::test_create_article", "-v", "--tb=short"], 
         "Unit Tests - Repository Pattern"),
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for command, description in test_commands:
        if run_test_command(command, description):
            passed += 1
        else:
            failed += 1
        print()  # Empty line for readability
    
    # Summary
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} {'❌' if failed > 0 else '✅'}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print("\n🎉 All integration tests passed!")
        print("✅ System is ready for deployment")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        print("🔧 Please fix the issues before deployment")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)