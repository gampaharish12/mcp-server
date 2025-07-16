#!/usr/bin/env python3
"""
FastAPI MCP Server for API Token Data Retrieval
This server provides endpoints to retrieve data from external APIs
using token authentication with display name and keyword parameters.
"""

import logging
import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager

# API Configuration
API_ENDPOINT = "https://apidata.globaldata.com/GlobalDataSocialMedia/api/content/GetOverallData"
API_TOKEN_ID = "jSUS2ZMHsdRF7seneARayXzrs2H5pqwdeU4qwSxDq="

# Validate configuration
if not API_ENDPOINT:
    raise ValueError("API_ENDPOINT must be configured")
if not API_TOKEN_ID:
    raise ValueError("API_TOKEN_ID must be configured")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FastMCP")

# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    global http_client
    
    # Startup
    logger.info("Starting FastMCP server...")
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )
    logger.info("HTTP client initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastMCP server...")
    if http_client:
        await http_client.aclose()
    logger.info("HTTP client closed")

class FastMCP(FastAPI):
    """Extended FastAPI class with MCP-specific functionality"""
    
    def __init__(self, name: str, host: str = "127.0.0.1", port: int = 5000, timeout: int = 30):
        super().__init__(
            title=name,
            description="Model Context Protocol Server for API Token Data Retrieval",
            version="1.0.0",
            lifespan=lifespan
        )
        self._host = host
        self._port = port
        self._timeout = timeout
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging middleware
        @self.middleware("http")
        async def log_requests(request, call_next):
            logger.info(f"Request: {request.method} {request.url}")
            try:
                response = await call_next(request)
                logger.info(f"Response status: {response.status_code}")
                return response
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                raise
    
    def run(self):
        """Run the FastAPI server"""
        logger.info(f"Starting server on {self._host}:{self._port}")
        uvicorn.run(
            self,
            host=self._host,
            port=self._port,
            timeout_keep_alive=self._timeout,
            log_level="info"
        )

# Initialize the FastAPI app
mcp = FastMCP(name="MCP Token Data Server", host="127.0.0.1", port=5000, timeout=30)

@mcp.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "name": "MCP Token Data Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "overall_data": "/overall_data",
            "health": "/health",
            "token_info": "/token_info"
        }
    }

@mcp.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-07-16T00:00:00Z",
        "api_endpoint": API_ENDPOINT,
        "token_configured": bool(API_TOKEN_ID)
    }

@mcp.get("/overall_data")
async def get_overall_data(
    display_name: str = Query(..., description="Display name to search for", min_length=1),
    keyword: str = Query(..., description="Keyword to filter data", min_length=1),
    provider: Optional[str] = Query(None, description="Optional provider filter"),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=100)
):
    """
    Fetch filtered data from the external API using token authentication.
    
    Args:
        display_name: The display name to search for
        keyword: The keyword to filter data
        provider: Optional provider filter
        limit: Maximum number of results to return
    
    Returns:
        JSON response with filtered data
    """
    if not http_client:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {API_TOKEN_ID}",
        "Content-Type": "application/json",
        "User-Agent": "FastMCP/1.0.0"
    }
    
    # Prepare query parameters
    params = {
        "display_name": display_name,
        "keyword": keyword,
        "limit": limit
    }
    
    if provider:
        params["provider"] = provider
    
    try:
        logger.info(f"Making API request to {API_ENDPOINT} with params: {params}")
        
        response = await http_client.get(
            API_ENDPOINT,
            headers=headers,
            params=params
        )
        
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"API request successful, received {len(data) if isinstance(data, list) else 'unknown'} items")
        
        return {
            "status": "success",
            "request_params": {
                "display_name": display_name,
                "keyword": keyword,
                "provider": provider,
                "limit": limit
            },
            "results": data,
            "total_results": len(data) if isinstance(data, list) else 1
        }
    
    except httpx.HTTPStatusError as exc:
        error_detail = f"API responded with status {exc.response.status_code}"
        try:
            error_data = exc.response.json()
            error_detail += f": {error_data.get('message', exc.response.text)}"
        except:
            error_detail += f": {exc.response.text}"
        
        logger.error(f"API error: {error_detail}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=error_detail
        )
    
    except httpx.TimeoutException:
        logger.error("API request timed out")
        raise HTTPException(
            status_code=504,
            detail="API request timed out"
        )
    
    except httpx.RequestError as exc:
        logger.error(f"Request error: {str(exc)}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to API: {str(exc)}"
        )
    
    except Exception as e:
        logger.exception("Unexpected error during API request")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@mcp.get("/token_info")
async def get_token_info():
    """
    Get information about the configured API token (without exposing the token itself).
    """
    return {
        "token_configured": bool(API_TOKEN_ID),
        "token_length": len(API_TOKEN_ID) if API_TOKEN_ID else 0,
        "token_prefix": API_TOKEN_ID[:8] + "..." if API_TOKEN_ID and len(API_TOKEN_ID) > 8 else None,
        "api_endpoint": API_ENDPOINT
    }

@mcp.post("/search_tokens")
async def search_tokens(
    display_name: str = Query(..., description="Display name to search for"),
    keyword: str = Query(..., description="Keyword to filter results"),
    provider: Optional[str] = Query(None, description="Provider filter"),
    limit: int = Query(10, description="Maximum results", ge=1, le=100)
):
    """
    Alternative endpoint for token search with POST method.
    Useful for more complex search operations.
    """
    # This would typically call the same logic as get_overall_data
    # but could be extended for more complex search functionality
    return await get_overall_data(
        display_name=display_name,
        keyword=keyword,
        provider=provider,
        limit=limit
    )

@mcp.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": "2025-07-16T00:00:00Z"
            }
        }
    )

@mcp.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "status_code": 500,
                "message": "Internal server error",
                "timestamp": "2025-07-16T00:00:00Z"
            }
        }
    )

if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise