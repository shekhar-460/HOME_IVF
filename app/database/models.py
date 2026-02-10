"""
SQLAlchemy database models with multilingual support
"""
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Conversation(Base):
    """Conversation model"""
    __tablename__ = "conversations"
    
    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    status = Column(String(20), default='active')  # 'active', 'escalated', 'closed'
    language = Column(String(10), default='en')  # 'en' or 'hi'
    conversation_metadata = Column('metadata', JSON, default={})  # Using 'metadata' as column name in DB
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    """Message model"""
    __tablename__ = "messages"
    
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.conversation_id'), nullable=False, index=True)
    sender_type = Column(String(20), nullable=False)  # 'patient' or 'bot'
    content = Column(Text, nullable=False)
    language = Column(String(10), default='en')  # Language of the message
    intent = Column(String(50), nullable=True)
    entities = Column(JSON, default=[])
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FAQ(Base):
    """FAQ model with multilingual support"""
    __tablename__ = "faqs"
    
    faq_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    question_hi = Column(Text, nullable=True)  # Hindi translation
    answer_hi = Column(Text, nullable=True)  # Hindi translation
    category = Column(String(50), nullable=True, index=True)
    tags = Column(ARRAY(String), default=[])
    embedding = Column(Text)  # JSON string of embedding vector
    view_count = Column(Integer, default=0)
    helpful_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Article(Base):
    """Educational article model with multilingual support"""
    __tablename__ = "articles"
    
    article_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    title_hi = Column(String(255), nullable=True)  # Hindi translation
    content_hi = Column(Text, nullable=True)  # Hindi translation
    category = Column(String(50), nullable=True)
    tags = Column(ARRAY(String), default=[])
    embedding = Column(Text)  # JSON string of embedding vector
    view_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Escalation(Base):
    """Escalation model"""
    __tablename__ = "escalations"
    
    escalation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.conversation_id'), nullable=False)
    reason = Column(String(100), nullable=False)
    urgency = Column(String(20), nullable=False)  # 'low', 'medium', 'high', 'urgent'
    counsellor_id = Column(UUID(as_uuid=True), nullable=True)
    status = Column(String(20), default='pending')  # 'pending', 'assigned', 'resolved'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
