"""Trace visualization API routes."""

from typing import Optional

import requests as http_requests
from fastapi import APIRouter, Header, Query
from fastapi.responses import JSONResponse

from app.server.config import (
    LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    LANGFLOW_BASE_URL, LANGFLOW_API_KEY, DEBUG_MODE, DEBUG_DIR,
)
from app.server.routers.trace_visualization.trace_fetcher import (
    get_flow_config, list_recent_traces,
)

router = APIRouter()


@router.get("/api/config", tags=["Configuration"])
def get_config():
    """Return client configuration including pre-configured API key."""
    return {"langflow_api_key": LANGFLOW_API_KEY}


def resolve_langflow_user_id(langflow_base_url: str, langflow_api_key: str) -> Optional[str]:
    """Resolve a LangFlow API key to a user ID via the whoami endpoint."""
    try:
        response = http_requests.get(
            f"{langflow_base_url}/api/v1/users/whoami",
            headers={"x-api-key": langflow_api_key},
        )
        response.raise_for_status()
        return response.json().get("id")
    except Exception:
        return None


@router.get("/api/refresh", tags=["Trace Visualization"])
def refresh(
    trace_id: Optional[str] = None,
    x_langflow_api_key: Optional[str] = Header(default=None),
):
    """Fetch trace and return flowConfig JSON."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return JSONResponse(status_code=500, content={
            "error": "LangFuse credentials not configured",
            "message": "Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        })

    if not x_langflow_api_key:
        return JSONResponse(status_code=400, content={
            "error": "LangFlow API key required",
            "message": "Please enter your LangFlow API key"
        })

    user_id = resolve_langflow_user_id(LANGFLOW_BASE_URL, x_langflow_api_key)
    if not user_id:
        return JSONResponse(status_code=401, content={
            "error": "Invalid API key",
            "message": "The provided LangFlow API key is invalid"
        })

    try:
        flow_config = get_flow_config(
            langfuse_base_url=LANGFUSE_BASE_URL,
            langfuse_public_key=LANGFUSE_PUBLIC_KEY,
            langfuse_secret_key=LANGFUSE_SECRET_KEY,
            langflow_base_url=LANGFLOW_BASE_URL,
            langflow_api_key=x_langflow_api_key,
            trace_id=trace_id,
            debug_mode=DEBUG_MODE,
            debug_dir=DEBUG_DIR,
            user_id=user_id,
        )
        return flow_config
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "Failed to fetch flow config",
            "message": str(e)
        })


@router.get("/api/traces", tags=["Trace Visualization"])
def traces(
    limit: int = Query(default=15, ge=1, le=50),
    page: int = Query(default=1, ge=1),
    session_id: Optional[str] = None,
    x_langflow_api_key: Optional[str] = Header(default=None),
):
    """List recent traces with pagination."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return JSONResponse(status_code=500, content={
            "error": "LangFuse credentials not configured",
            "message": "Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        })

    if not x_langflow_api_key:
        return JSONResponse(status_code=400, content={
            "error": "LangFlow API key required",
            "message": "Please enter your LangFlow API key"
        })

    user_id = resolve_langflow_user_id(LANGFLOW_BASE_URL, x_langflow_api_key)
    if not user_id:
        return JSONResponse(status_code=401, content={
            "error": "Invalid API key",
            "message": "The provided LangFlow API key is invalid"
        })

    try:
        result = list_recent_traces(
            langfuse_base_url=LANGFUSE_BASE_URL,
            langfuse_public_key=LANGFUSE_PUBLIC_KEY,
            langfuse_secret_key=LANGFUSE_SECRET_KEY,
            limit=limit,
            page=page,
            session_id=session_id,
            user_id=user_id,
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "Failed to fetch traces",
            "message": str(e)
        })
