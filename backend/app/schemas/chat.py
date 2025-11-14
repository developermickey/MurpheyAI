from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class MessageRequest(BaseModel):
    content: str
    conversation_id: Optional[int] = None
    model: Optional[str] = "small"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True
    system_prompt: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    tokens_used: int
    created_at: datetime


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: Optional[str]
    model_used: str
    total_tokens: int
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []
    
    class Config:
        from_attributes = True


class ChatSettings(BaseModel):
    model: str = "small"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    tone: Optional[str] = "balanced"  # professional, casual, creative
    creativity: Optional[float] = 0.7  # 0.0 to 1.0


class UsageStats(BaseModel):
    total_tokens: int
    tokens_today: int
    tokens_this_month: int
    credits_remaining: float
    conversations_count: int

