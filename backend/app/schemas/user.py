from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    preferred_model: Optional[str] = None
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    credits: float
    total_tokens_used: int
    created_at: datetime
    preferred_model: str
    default_temperature: float
    default_max_tokens: int
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None

