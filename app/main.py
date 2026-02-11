"""
Main FastAPI application
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import uvicorn

from app.config import settings
from app.api.routes import chat, health, admin, engagement, translate
from app.database.connection import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Suppress uvicorn warnings for invalid HTTP requests
# These are typically from browsers making malformed requests (favicon, HTTP/2 upgrades, etc.)
# and are harmless - they occur before requests reach FastAPI
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.WARNING)  # Only show warnings and errors

# Create FastAPI app with enhanced OpenAPI documentation
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## AI Engagement Tools (Home IVF)
    
    A comprehensive conversational AI system providing educational support to patients undergoing IVF treatment.
    Supports **Hindi** and **English** languages with intelligent fallback to AI-generated responses.
    
    ### Key Features
    
    * 🌐 **Multilingual Support**: Automatic language detection (Hindi/English)
    * 💬 **Natural Language Understanding**: Intent classification and semantic search
    * 📚 **Knowledge Base**: FAQ and article search with semantic similarity
    * 🤖 **AI Fallback**: Medgemma-4b-it model for unanswered questions
    * 🎯 **Smart Escalation**: Automatic routing to human counsellors
    * 💡 **Interactive**: Follow-up questions and proactive suggestions
    * ✨ **Formatted Responses**: Clean, structured output (150 word limit)
    
    ### API Endpoints
    
    * **Chat**: Send messages, create conversations, get history
    * **Health**: Service health and readiness checks
    * **Admin**: Analytics, FAQ/article management
    
    ### Authentication
    
    Currently no authentication required for development. Production deployments should implement JWT or API key authentication.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "chat",
            "description": "Chat endpoints for patient interactions. Send messages, create conversations, and manage chat sessions."
        },
        {
            "name": "health",
            "description": "Health check endpoints for monitoring service status and readiness."
        },
        {
            "name": "admin",
            "description": "Admin endpoints for managing knowledge base, viewing analytics, and system administration."
        },
        {
            "name": "engagement",
            "description": "AI-driven engagement tools: fertility readiness calculator, hormonal predictor, visual wellness (exploratory), treatment pathway recommender, and Home IVF eligibility checker."
        }
    ],
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# CORS middleware - Allow frontend from any origin (different devices, ports 3000/4200, etc.)
# Using allow_origins=["*"] with allow_credentials=False so the browser always accepts the
# response (our frontend does not send cookies to the API).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def ensure_cors_headers(request: Request, call_next):
    """Ensure every response has CORS headers so frontend from any origin works."""
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# Middleware to handle malformed requests gracefully
@app.middleware("http")
async def handle_malformed_requests(request: Request, call_next):
    """Handle malformed HTTP requests gracefully"""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log the error but don't crash
        if "Invalid HTTP request" in str(e) or "Invalid" in str(type(e).__name__):
            logger.debug(f"Bad request from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request"}
            )
        raise

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
app.include_router(engagement.router, prefix="/api/v1/engagement", tags=["engagement"])
app.include_router(translate.router, prefix="/api/v1", tags=["translate"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Languages: {', '.join(settings.SUPPORTED_LANGUAGES)}")
    
    # CRITICAL: Cleanup any leftover GPU memory from previous runs
    # This ensures clean state on restart
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Cleaning up any leftover GPU memory from previous runs...")
            from app.services.knowledge_engine import MedgemmaModelManager
            
            # Force cleanup of model manager singleton
            manager = MedgemmaModelManager()
            manager.cleanup(force=True)
            
            # Aggressive CUDA cache clear
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    for _ in range(5):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                    try:
                        torch.cuda.reset_peak_memory_stats(i)
                    except:
                        pass
            
            # Log initial GPU memory state
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i} startup state: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            logger.info("GPU memory cleanup on startup completed")
    except Exception as e:
        logger.warning(f"GPU cleanup on startup failed (non-critical): {e}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database ready")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        # Don't fail startup if DB is not available (for development)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down")
    
    # Cleanup Medgemma model to release GPU memory
    try:
        from app.services.knowledge_engine import MedgemmaModelManager
        model_manager = MedgemmaModelManager()
        model_manager.cleanup(force=True)
        logger.info("GPU memory released on shutdown")
    except Exception as e:
        logger.warning(f"Failed to cleanup model on shutdown: {e}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Invalid request: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get(
    "/",
    tags=["health"],
    summary="Root endpoint",
    description="Get service information and status",
    response_description="Service metadata including name, version, status, and supported languages"
)
async def root():
    """
    Root endpoint providing service information.
    
    Returns basic service metadata including:
    - Service name and version
    - Current status
    - Supported languages
    - API documentation URL
    """
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "docs": "/docs",
        "api_version": "v1"
    }


@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404s"""
    from fastapi.responses import Response
    return Response(status_code=204)  # No Content
