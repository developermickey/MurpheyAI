from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
import redis
from app.core.config import settings

# PostgreSQL
engine = create_engine(settings.POSTGRES_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# MongoDB
mongo_client = MongoClient(settings.MONGODB_URL)
mongodb = mongo_client.get_database()

# Redis
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def get_db():
    """Get PostgreSQL database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_mongodb():
    """Get MongoDB database."""
    return mongodb


def get_redis():
    """Get Redis client."""
    return redis_client

