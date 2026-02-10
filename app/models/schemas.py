"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class ChatMessageRequest(BaseModel):
    """Request model for chat message"""
    conversation_id: Optional[UUID] = Field(None, description="Existing conversation ID. If not provided, a new conversation is created.")
    patient_id: UUID = Field(..., description="Unique patient identifier (UUID)")
    message: str = Field(..., min_length=1, max_length=1000, description="Patient's message/question", example="What is IVF?")
    language: Optional[str] = Field(default="en", description="Language code: 'en' (English) or 'hi' (Hindi). Auto-detected if not provided.")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context for the conversation")
    image_base64: Optional[str] = Field(default=None, description="Optional image (base64) for multimodal MedGemma Q&A (e.g. scan, photo)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "What is IVF?",
                "language": "en"
            }
        }


class ConversationRequest(BaseModel):
    """Request model for new conversation or sending message to existing conversation"""
    patient_id: UUID = Field(..., description="Unique patient identifier (UUID)")
    conversation_id: Optional[UUID] = Field(None, description="Existing conversation ID. If provided, message is added to that conversation.")
    initial_message: Optional[str] = Field(None, description="Initial message to start the conversation", example="नमस्ते, मैं आईवीएफ के बारे में जानना चाहती हूं")
    language: Optional[str] = Field(default="en", description="Language code: 'en' (English) or 'hi' (Hindi). Auto-detected if not provided.")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata for the conversation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "123e4567-e89b-12d3-a456-426614174000",
                "initial_message": "What is IVF?",
                "language": "en"
            }
        }


class Intent(BaseModel):
    """Intent classification result"""
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_probabilities: Optional[Dict[str, float]] = {}
    entities: Optional[List[Dict[str, Any]]] = []


class SuggestedAction(BaseModel):
    """Suggested action for user"""
    type: str  # 'link', 'button', 'quick_reply'
    label: str
    action: Optional[str] = None
    url: Optional[str] = None
    data: Optional[Dict[str, Any]] = {}


class RelatedContent(BaseModel):
    """Related content item"""
    title: str
    id: str
    type: Optional[str] = "article"  # 'article', 'faq'
    url: Optional[str] = None


class BotResponse(BaseModel):
    """Bot response model"""
    text: str
    type: str = "text"  # 'text', 'card', 'carousel', 'quick_replies'
    language: str = "en"
    suggested_actions: Optional[List[SuggestedAction]] = []
    related_content: Optional[List[RelatedContent]] = []
    confidence: Optional[float] = None
    sources: Optional[List[str]] = []


class ChatMessageResponse(BaseModel):
    """Response model for chat message"""
    conversation_id: UUID
    message_id: UUID
    response: BotResponse
    intent: Intent
    escalation: Dict[str, Any]
    timestamp: datetime


class ConversationResponse(BaseModel):
    """Response model for conversation"""
    conversation_id: UUID
    patient_id: UUID
    status: str
    language: str
    created_at: datetime
    response: Optional[BotResponse] = None


class ConversationHistory(BaseModel):
    """Conversation history model"""
    conversation_id: UUID
    patient_id: UUID
    messages: List[Dict[str, Any]]
    status: str
    language: str
    created_at: datetime
    last_activity: datetime


class EscalationRequest(BaseModel):
    """Escalation request model"""
    conversation_id: UUID
    reason: str
    urgency: str = Field(..., pattern="^(low|medium|high|urgent)$")
    patient_id: UUID


class EscalationResponse(BaseModel):
    """Escalation response model"""
    escalation_id: UUID
    conversation_id: UUID
    status: str
    urgency: str
    estimated_wait_time: Optional[int] = None
    message: str
