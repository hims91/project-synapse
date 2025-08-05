"""
Admin endpoints for Project Synapse.
Used for system management and component registration.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import secrets
import hashlib
from datetime import datetime

router = APIRouter(prefix="/v1/admin", tags=["admin"])

# In-memory storage for demo (use database in production)
COMPONENT_API_KEYS = {}

@router.post("/setup")
async def setup_system():
    """Initialize system and generate component API keys."""
    
    # Generate API keys for each component
    components = ["dendrites", "neurons", "sensory", "spinal"]
    
    for component in components:
        api_key = f"{component}_{secrets.token_urlsafe(32)}"
        COMPONENT_API_KEYS[component] = {
            "api_key": api_key,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
    
    return {
        "status": "success",
        "message": "System initialized successfully",
        "api_keys": {
            component: data["api_key"] 
            for component, data in COMPONENT_API_KEYS.items()
        }
    }

@router.get("/components/{component}/api-key")
async def get_component_api_key(component: str):
    """Get API key for a specific component."""
    
    if component not in COMPONENT_API_KEYS:
        raise HTTPException(status_code=404, detail="Component not found")
    
    return {
        "component": component,
        "api_key": COMPONENT_API_KEYS[component]["api_key"],
        "status": COMPONENT_API_KEYS[component]["status"]
    }

@router.get("/system/status")
async def get_system_status():
    """Get overall system status."""
    
    return {
        "status": "healthy",
        "components": {
            "central_cortex": "healthy",
            "dendrites": "healthy" if "dendrites" in COMPONENT_API_KEYS else "not_configured",
            "neurons": "healthy" if "neurons" in COMPONENT_API_KEYS else "not_configured", 
            "sensory_neurons": "healthy" if "sensory" in COMPONENT_API_KEYS else "not_configured",
            "spinal_cord": "healthy" if "spinal" in COMPONENT_API_KEYS else "not_configured",
            "frontend": "healthy"
        },
        "total_components": 6,
        "configured_components": len(COMPONENT_API_KEYS),
        "timestamp": datetime.utcnow().isoformat()
    }