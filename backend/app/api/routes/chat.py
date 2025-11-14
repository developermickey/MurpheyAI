from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.schemas.chat import MessageRequest, ConversationResponse, ConversationCreate, UsageStats
from app.services.chat_service import ChatService
from app.services.model_service import model_service
from typing import List
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    conversation = ChatService.create_conversation(
        db, current_user.id, conversation_data.title
    )
    return conversation


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's conversations."""
    conversations = ChatService.get_conversations(db, current_user.id, limit, offset)
    return conversations


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific conversation."""
    conversation = ChatService.get_conversation(db, conversation_id, current_user.id)
    return conversation


@router.post("/message")
async def send_message(
    message: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message (non-streaming)."""
    try:
        response_chunks = []
        async for chunk in ChatService.process_message(
            db=db,
            user=current_user,
            content=message.content,
            conversation_id=message.conversation_id,
            model=message.model or "small",
            temperature=message.temperature or 0.7,
            max_tokens=message.max_tokens or 2048,
            stream=False
        ):
            response_chunks.append(chunk)
        
        return {"response": "".join(response_chunks)}
    except ValueError as e:
        return {"error": str(e)}


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket, db: Session = Depends(get_db)):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()
    
    try:
        # In production, authenticate via token in query params
        # For now, accept all connections
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message and stream response
            async for chunk in ChatService.process_message(
                db=db,
                user=None,  # Should get from auth
                content=message_data.get("content", ""),
                conversation_id=message_data.get("conversation_id"),
                model=message_data.get("model", "small"),
                temperature=message_data.get("temperature", 0.7),
                max_tokens=message_data.get("max_tokens", 2048),
                stream=True
            ):
                await websocket.send_text(json.dumps({"chunk": chunk}))
            
            await websocket.send_text(json.dumps({"done": True}))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's usage statistics."""
    from datetime import datetime, timedelta
    from sqlalchemy import func
    
    from app.models.conversation import Conversation
    
    today = datetime.utcnow().date()
    month_start = datetime.utcnow().replace(day=1).date()
    
    conversations_count = db.query(func.count(Conversation.id)).filter(
        Conversation.user_id == current_user.id
    ).scalar()
    
    return UsageStats(
        total_tokens=current_user.total_tokens_used,
        tokens_today=0,  # Calculate from messages
        tokens_this_month=0,  # Calculate from messages
        credits_remaining=current_user.credits,
        conversations_count=conversations_count or 0
    )

