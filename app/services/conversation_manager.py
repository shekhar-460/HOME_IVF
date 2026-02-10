"""
Conversation Manager - Handles conversation sessions and context
"""
from typing import Optional, Dict, List
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.database.models import Conversation, Message
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manage conversation sessions and context"""
    
    def __init__(self, db: Session):
        self.db = db
        self.session_timeout = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
    
    def create_conversation(
        self,
        patient_id: UUID,
        language: str = "en",
        metadata: Optional[Dict] = None
    ) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(
            conversation_id=uuid4(),
            patient_id=patient_id,
            status='active',
            language=language,
            conversation_metadata=metadata or {}
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        logger.info(f"Created conversation {conversation.conversation_id} for patient {patient_id}")
        return conversation
    
    def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
    
    def get_active_conversation(
        self,
        patient_id: UUID,
        language: Optional[str] = None
    ) -> Optional[Conversation]:
        """Get active conversation for patient"""
        query = self.db.query(Conversation).filter(
            Conversation.patient_id == patient_id,
            Conversation.status == 'active'
        )
        
        if language:
            query = query.filter(Conversation.language == language)
        
        conversation = query.order_by(Conversation.last_activity.desc()).first()
        
        # Check if conversation has timed out (use timezone-aware UTC to match DB)
        if conversation:
            now_utc = datetime.now(timezone.utc)
            last = conversation.last_activity
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if now_utc - last > self.session_timeout:
                self.close_conversation(conversation.conversation_id)
                return None
        
        return conversation
    
    def add_message(
        self,
        conversation_id: UUID,
        sender_type: str,
        content: str,
        language: str = "en",
        intent: Optional[str] = None,
        entities: Optional[List[Dict]] = None,
        confidence_score: Optional[float] = None
    ) -> Message:
        """Add message to conversation"""
        message = Message(
            message_id=uuid4(),
            conversation_id=conversation_id,
            sender_type=sender_type,
            content=content,
            language=language,
            intent=intent,
            entities=entities or [],
            confidence_score=confidence_score
        )
        self.db.add(message)
        
        # Update conversation last activity
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.last_activity = datetime.now(timezone.utc)
        
        self.db.commit()
        self.db.refresh(message)
        return message
    
    def get_conversation_history(
        self,
        conversation_id: UUID,
        limit: int = None
    ) -> List[Message]:
        """Get conversation history"""
        query = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.asc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def update_conversation_language(
        self,
        conversation_id: UUID,
        language: str
    ) -> bool:
        """Update conversation language"""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.language = language
            self.db.commit()
            return True
        return False
    
    def close_conversation(self, conversation_id: UUID) -> bool:
        """Close a conversation"""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.status = 'closed'
            self.db.commit()
            return True
        return False
    
    def escalate_conversation(self, conversation_id: UUID) -> bool:
        """Mark conversation as escalated"""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.status = 'escalated'
            self.db.commit()
            return True
        return False
