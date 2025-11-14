from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.dependencies import get_current_admin_user
from app.models.user import User
from app.schemas.user import UserResponse
from typing import List
from sqlalchemy import func

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)."""
    users = db.query(User).offset(offset).limit(limit).all()
    return users


@router.get("/stats")
async def get_admin_stats(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get platform statistics (admin only)."""
    from app.models.conversation import Conversation, Message
    
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    total_conversations = db.query(func.count(Conversation.id)).scalar()
    total_messages = db.query(func.count(Message.id)).scalar()
    total_tokens = db.query(func.sum(User.total_tokens_used)).scalar() or 0
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_tokens_used": total_tokens
    }

