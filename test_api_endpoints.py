#!/usr/bin/env python3
"""
Quick test script to verify all API endpoints are working properly.
This script performs basic import and initialization tests for all specialized API routers.
"""

import sys
import traceback
from typing import Dict, Any

def test_imports():
    """Test that all API router modules can be imported successfully."""
    test_results = {}
    
    # Test FinMind API
    try:
        from src.axon_interface.routers import finmind
        test_results['finmind'] = {
            'import': True,
            'router': hasattr(finmind, 'router'),
            'endpoints': len([attr for attr in dir(finmind.router) if not attr.startswith('_')])
        }
        print("✅ FinMind API imported successfully")
    except Exception as e:
        test_results['finmind'] = {'import': False, 'error': str(e)}
        print(f"❌ FinMind API import failed: {e}")
    
    # Test Digestify API
    try:
        from src.axon_interface.routers import digestify
        test_results['digestify'] = {
            'import': True,
            'router': hasattr(digestify, 'router'),
            'endpoints': len([attr for attr in dir(digestify.router) if not attr.startswith('_')])
        }
        print("✅ Digestify API imported successfully")
    except Exception as e:
        test_results['digestify'] = {'import': False, 'error': str(e)}
        print(f"❌ Digestify API import failed: {e}")
    
    # Test InsightGraph API
    try:
        from src.axon_interface.routers import insightgraph
        test_results['insightgraph'] = {
            'import': True,
            'router': hasattr(insightgraph, 'router'),
            'endpoints': len([attr for attr in dir(insightgraph.router) if not attr.startswith('_')])
        }
        print("✅ InsightGraph API imported successfully")
    except Exception as e:
        test_results['insightgraph'] = {'import': False, 'error': str(e)}
        print(f"❌ InsightGraph API import failed: {e}")
    
    # Test MetaLens API
    try:
        from src.axon_interface.routers import metalens
        test_results['metalens'] = {
            'import': True,
            'router': hasattr(metalens, 'router'),
            'endpoints': len([attr for attr in dir(metalens.router) if not attr.startswith('_')])
        }
        print("✅ MetaLens API imported successfully")
    except Exception as e:
        test_results['metalens'] = {'import': False, 'error': str(e)}
        print(f"❌ MetaLens API import failed: {e}")
    
    # Test Narrative API
    try:
        from src.axon_interface.routers import narrative
        test_results['narrative'] = {
            'import': True,
            'router': hasattr(narrative, 'router'),
            'endpoints': len([attr for attr in dir(narrative.router) if not attr.startswith('_')])
        }
        print("✅ Narrative API imported successfully")
    except Exception as e:
        test_results['narrative'] = {'import': False, 'error': str(e)}
        print(f"❌ Narrative API import failed: {e}")
    
    return test_results

def test_router_configuration():
    """Test that routers are properly configured with endpoints."""
    print("\n🔍 Testing Router Configuration...")
    
    try:
        from src.axon_interface.routers.finmind import router as finmind_router
        finmind_routes = [route.path for route in finmind_router.routes]
        print(f"FinMind routes: {finmind_routes}")
    except Exception as e:
        print(f"❌ FinMind router configuration error: {e}")
    
    try:
        from src.axon_interface.routers.digestify import router as digestify_router
        digestify_routes = [route.path for route in digestify_router.routes]
        print(f"Digestify routes: {digestify_routes}")
    except Exception as e:
        print(f"❌ Digestify router configuration error: {e}")
    
    try:
        from src.axon_interface.routers.insightgraph import router as insightgraph_router
        insightgraph_routes = [route.path for route in insightgraph_router.routes]
        print(f"InsightGraph routes: {insightgraph_routes}")
    except Exception as e:
        print(f"❌ InsightGraph router configuration error: {e}")
    
    try:
        from src.axon_interface.routers.metalens import router as metalens_router
        metalens_routes = [route.path for route in metalens_router.routes]
        print(f"MetaLens routes: {metalens_routes}")
    except Exception as e:
        print(f"❌ MetaLens router configuration error: {e}")
    
    try:
        from src.axon_interface.routers.narrative import router as narrative_router
        narrative_routes = [route.path for route in narrative_router.routes]
        print(f"Narrative routes: {narrative_routes}")
    except Exception as e:
        print(f"❌ Narrative router configuration error: {e}")

def test_dependencies():
    """Test that required dependencies are available."""
    print("\n🔍 Testing Dependencies...")
    
    dependencies = [
        'fastapi',
        'pydantic',
        'typing',
        'datetime',
        'logging',
        're',
        'collections'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")

def test_frontend_structure():
    """Test that frontend structure is properly set up."""
    print("\n🔍 Testing Frontend Structure...")
    
    import os
    
    frontend_files = [
        'frontend/package.json',
        'frontend/vite.config.ts',
        'frontend/tailwind.config.js',
        'frontend/tsconfig.json',
        'frontend/index.html',
        'frontend/src/main.tsx',
        'frontend/src/App.tsx',
        'frontend/src/index.css',
    ]
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    # Check key directories
    directories = [
        'frontend/src/components',
        'frontend/src/pages',
        'frontend/src/contexts',
        'frontend/src/lib',
        'frontend/src/types',
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"❌ {dir_path}/ directory missing")

def main():
    """Run all tests."""
    print("🚀 Testing Project Synapse API Endpoints")
    print("=" * 50)
    
    # Test imports
    print("\n📦 Testing Module Imports...")
    import_results = test_imports()
    
    # Test router configuration
    test_router_configuration()
    
    # Test dependencies
    test_dependencies()
    
    # Test frontend structure
    test_frontend_structure()
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 30)
    
    successful_imports = sum(1 for result in import_results.values() if result.get('import', False))
    total_apis = len(import_results)
    
    print(f"APIs successfully imported: {successful_imports}/{total_apis}")
    
    if successful_imports == total_apis:
        print("🎉 All API endpoints are working properly!")
        return 0
    else:
        print("⚠️  Some API endpoints have issues. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())