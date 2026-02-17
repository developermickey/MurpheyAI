from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "MurpheyAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:3001"]

    # Database
    # Default to a local SQLite file for easy local testing when POSTGRES_URL
    # is not provided via environment. In production, set POSTGRES_URL to a
    # PostgreSQL connection string in the environment (.env).
    POSTGRES_URL: str = "sqlite:///./dev.db"
    MONGODB_URL: str = "mongodb://localhost:27017/murpheyai"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Vector Database
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "murpheyai-embeddings"

    # Model Configuration
    # Empty means use Hugging Face model directly, or "./models" for custom models
    MODEL_PATH: str = ""
    # Model name: "gpt2" (Hugging Face) or custom model name from ./models/
    MODEL_NAME: str = "gpt2"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # GPU Configuration
    GPU_DEVICE: str = "cuda:0"
    BATCH_SIZE: int = 1
    MAX_BATCH_SIZE: int = 32

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".md"]

    # JWT
    JWT_SECRET_KEY: str = "your-jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"

    # OAuth
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    ENABLE_METRICS: bool = True

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
