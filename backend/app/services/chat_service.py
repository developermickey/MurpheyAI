from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.model_service import model_service
from app.core.security import detect_jailbreak_attempt, sanitize_input
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ChatService:
    @staticmethod
    def create_conversation(db: Session, user_id: int, title: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            user_id=user_id,
            title=title or "New Conversation"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    
    @staticmethod
    def get_conversations(db: Session, user_id: int, limit: int = 50, offset: int = 0) -> List[Conversation]:
        """Get user's conversations."""
        return db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(
            Conversation.updated_at.desc()
        ).offset(offset).limit(limit).all()
    
    @staticmethod
    def get_conversation(db: Session, conversation_id: int, user_id: int) -> Conversation:
        """Get a specific conversation."""
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        if not conversation:
            raise ValueError("Conversation not found")
        return conversation
    
    @staticmethod
    def add_message(
        db: Session,
        conversation_id: int,
        role: str,
        content: str,
        tokens_used: int = 0
    ) -> Message:
        """Add a message to a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_used=tokens_used
        )
        db.add(message)
        
        # Update conversation token count
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if conversation:
            conversation.total_tokens += tokens_used
            conversation.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        return message
    
    @staticmethod
    async def process_message(
        db: Session,
        user: User,
        content: str,
        conversation_id: Optional[int] = None,
        model: str = "small",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = True
    ):
        """Process a user message and generate response."""
        # Sanitize input
        content = sanitize_input(content)
        
        # Check for jailbreak attempts
        if detect_jailbreak_attempt(content):
            logger.warning(f"Jailbreak attempt detected from user {user.id}")
            return "I cannot fulfill this request as it may violate safety guidelines."
        
        # Get or create conversation
        if conversation_id:
            try:
                conversation = ChatService.get_conversation(db, conversation_id, user.id)
            except ValueError:
                conversation = ChatService.create_conversation(db, user.id)
        else:
            conversation = ChatService.create_conversation(db, user.id)
        
        # Count tokens
        input_tokens = model_service.count_tokens(content, model)
        
        # Check credits
        if user.credits < input_tokens * 0.001:  # Example pricing
            raise ValueError("Insufficient credits")
        
        # Save user message
        user_message = ChatService.add_message(
            db, conversation.id, "user", content, int(input_tokens)
        )
        
        # Update user token usage
        user.total_tokens_used += int(input_tokens)
        user.credits -= input_tokens * 0.001
        db.commit()
        
        # Generate response
        response_tokens = 0
        full_response = ""
        
        async for chunk in model_service.generate(
            content, model, temperature, max_tokens, stream
        ):
            full_response += chunk
            response_tokens += model_service.count_tokens(chunk, model)
            yield chunk
        
        # Save assistant message
        ChatService.add_message(
            db, conversation.id, "assistant", full_response, int(response_tokens)
        )
        
        # Update user credits
        user.total_tokens_used += int(response_tokens)
        user.credits -= response_tokens * 0.001
        db.commit()
        
        return conversation.id

