"""
Configuration settings for AI Engagement Tools (Home IVF)
"""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "AI Engagement Tools (Home IVF)"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    # Uvicorn Workers (for production)
    UVICORN_WORKERS: int = Field(default=4, env="UVICORN_WORKERS")  # Number of worker processes
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi"]  # English, Hindi
    DEFAULT_LANGUAGE: str = "en"
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5433/patient_bot",
        env="DATABASE_URL"
    )
    # Database Connection Pool Settings
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")  # Number of connections to maintain
    DB_MAX_OVERFLOW: int = Field(default=10, env="DB_MAX_OVERFLOW")  # Additional connections beyond pool_size
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")  # Seconds to wait for connection
    DB_POOL_RECYCLE: int = Field(default=3600, env="DB_POOL_RECYCLE")  # Recycle connections after 1 hour
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Vector Database (Pinecone)
    PINECONE_API_KEY: str = Field(default="", env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(default="", env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = Field(default="ivf-knowledge-base", env="PINECONE_INDEX_NAME")
    
    # NLP Models
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )  # Multilingual model for Hindi and English
    
    INTENT_MODEL_PATH: str = Field(
        default="models/intent_classifier",
        env="INTENT_MODEL_PATH"
    )
    
    # Medgemma Configuration (Fallback AI Model)
    USE_MEDGEMMA: bool = Field(
        default=True,
        env="USE_MEDGEMMA"
    )  # Enable medgemma as fallback for unanswered questions
    
    MEDGEMMA_MODEL_NAME: str = Field(
        default="google/medgemma-4b-it",
        env="MEDGEMMA_MODEL_NAME"
    )  # Medgemma multimodal model name (HuggingFace model ID)
    
    MEDGEMMA_MODEL_PATH: str = Field(
        default="app/models/medgemma-4b-it",
        env="MEDGEMMA_MODEL_PATH"
    )  # Local path to medgemma model (if downloaded)
    
    USE_LOCAL_MEDGEMMA: bool = Field(
        default=True,
        env="USE_LOCAL_MEDGEMMA"
    )  # Use local model if available, otherwise download from HuggingFace
    
    # Translation Service
    USE_GOOGLE_TRANSLATE: bool = Field(default=True, env="USE_GOOGLE_TRANSLATE")
    GOOGLE_TRANSLATE_API_KEY: str = Field(default="", env="GOOGLE_TRANSLATE_API_KEY")
    
    # Appointment System
    APPOINTMENT_API_URL: str = Field(
        default="http://localhost:8001/api",
        env="APPOINTMENT_API_URL"
    )
    APPOINTMENT_API_KEY: str = Field(default="", env="APPOINTMENT_API_KEY")
    
    # Message Queue
    RABBITMQ_URL: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        env="RABBITMQ_URL"
    )
    
    # Security
    JWT_SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="JWT_SECRET_KEY"
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Rate Limiting (per user/IP)
    RATE_LIMIT_PER_MINUTE: int = Field(default=300, env="RATE_LIMIT_PER_MINUTE")  # Increased from 60
    RATE_LIMIT_PER_HOUR: int = Field(default=10000, env="RATE_LIMIT_PER_HOUR")  # Increased from 1000
    
    # Session Management
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_CONVERSATION_HISTORY: int = 50
    
    # Escalation
    ESCALATION_URGENCY_THRESHOLD: float = 3.0
    ESCALATION_COMPLEXITY_THRESHOLD: float = 2.0
    
    # Response Generation
    MAX_RESPONSE_LENGTH: int = 1000
    TOP_K_SEARCH_RESULTS: int = 5
    MIN_CONFIDENCE_SCORE: float = 0.6
    
    # Caching
    CACHE_TTL_FAQ: int = Field(default=3600, env="CACHE_TTL_FAQ")  # 1 hour
    CACHE_TTL_MEDGEMMA: int = Field(default=86400, env="CACHE_TTL_MEDGEMMA")  # 24 hours
    
    # Interactivity
    ENABLE_FOLLOWUPS: bool = Field(default=True, env="ENABLE_FOLLOWUPS")
    ENABLE_PROACTIVE_SUGGESTIONS: bool = Field(default=True, env="ENABLE_PROACTIVE_SUGGESTIONS")
    MAX_FOLLOWUP_QUESTIONS: int = Field(default=3, env="MAX_FOLLOWUP_QUESTIONS")
    
    # Performance Optimizations
    PRE_COMPUTE_EMBEDDINGS: bool = Field(default=False, env="PRE_COMPUTE_EMBEDDINGS")  # Pre-compute FAQ embeddings on startup
    
    # External Resources (professional help)
    HOMEIVF_WEBSITE_URL: str = Field(
        default="https://homeivf.com/",
        env="HOMEIVF_WEBSITE_URL"
    )  # HomeIVF website – book consultations, expert care
    HOMEIVF_PHONE: str = Field(
        default="+91-9958885250",
        env="HOMEIVF_PHONE"
    )  # HomeIVF contact number
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
