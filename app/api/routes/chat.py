"""
Chat endpoints for patient interactions
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
from typing import Optional
import json

from app.database.connection import get_db
from app.models.schemas import (
    ChatMessageRequest, ChatMessageResponse, ConversationRequest,
    ConversationResponse, ConversationHistory, EscalationRequest, EscalationResponse,
    Intent
)
from app.services.conversation_manager import ConversationManager
from app.services.intent_classifier import IntentClassifier
from app.services.knowledge_engine import KnowledgeEngine
from app.services.response_generator import ResponseGenerator
from app.services.escalation_manager import EscalationManager
from app.utils.language_detector import language_detector
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def get_conversation_manager(db: Session = Depends(get_db)) -> ConversationManager:
    """Get conversation manager instance"""
    return ConversationManager(db)


def get_intent_classifier() -> IntentClassifier:
    """Get intent classifier instance"""
    return IntentClassifier()


def get_knowledge_engine(db: Session = Depends(get_db)) -> KnowledgeEngine:
    """Get knowledge engine instance"""
    return KnowledgeEngine(db)


def get_escalation_manager(db: Session = Depends(get_db)) -> EscalationManager:
    """Get escalation manager instance"""
    return EscalationManager(db)


def get_response_generator(
    intent_classifier: IntentClassifier = Depends(get_intent_classifier),
    knowledge_engine: KnowledgeEngine = Depends(get_knowledge_engine),
    escalation_manager: EscalationManager = Depends(get_escalation_manager)
) -> ResponseGenerator:
    """Get response generator instance"""
    return ResponseGenerator(intent_classifier, knowledge_engine, escalation_manager)


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="Send a message to the bot",
    description="""
    Send a message to the bot and receive an intelligent response.
    
    The bot will:
    - Automatically detect language if not specified
    - Classify the intent of your message
    - Search the knowledge base for relevant FAQs/articles
    - Use AI fallback (Medgemma) if no good match found
    - Generate follow-up questions and suggestions
    - Return formatted, readable response (max 150 words for AI responses)
    
    **Language Detection**: If `language` is not provided, the bot automatically detects
    the language from the message content (supports Hindi and English).
    
    **Conversation Management**: If `conversation_id` is provided, the message is added to
    that conversation. Otherwise, a new conversation is created or an active one is used.
    """,
    response_description="Bot response with intent classification, suggested actions, and related content",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                        "message_id": "223e4567-e89b-12d3-a456-426614174001",
                        "response": {
                            "text": "IVF (In Vitro Fertilization) is a fertility treatment...",
                            "type": "text",
                            "language": "en",
                            "suggested_actions": [
                                {
                                    "type": "quick_reply",
                                    "label": "How long does the IVF process take?",
                                    "action": "faq_followup"
                                }
                            ],
                            "related_content": [],
                            "confidence": 0.85
                        },
                        "intent": {
                            "name": "faq_general",
                            "confidence": 0.85,
                            "all_probabilities": {"faq_general": 0.85},
                            "entities": []
                        },
                        "escalation": {"escalate": False},
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        404: {"description": "Conversation not found"},
        500: {"description": "Internal server error"}
    }
)
async def send_message(
    request: ChatMessageRequest,
    db: Session = Depends(get_db),
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
    response_generator: ResponseGenerator = Depends(get_response_generator)
):
    """Send a message to the bot"""
    try:
        # Detect language if not provided
        language = request.language or language_detector.detect_language(request.message)
        if not language_detector.validate_language(language):
            language = settings.DEFAULT_LANGUAGE
        
        # Get or create conversation
        conversation = None
        if request.conversation_id:
            conversation = conversation_manager.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            # Update language if changed
            if conversation.language != language:
                conversation_manager.update_conversation_language(request.conversation_id, language)
        else:
            # Get active conversation or create new
            conversation = conversation_manager.get_active_conversation(
                request.patient_id, language
            )
            if not conversation:
                conversation = conversation_manager.create_conversation(
                    request.patient_id, language, request.context
                )
        
        # Add patient message
        patient_message = conversation_manager.add_message(
            conversation.conversation_id,
            'patient',
            request.message,
            language
        )
        
        # Get conversation history for context
        history = conversation_manager.get_conversation_history(conversation.conversation_id, limit=10)
        conversation_context = {
            'history': [
                {
                    'sender': msg.sender_type,
                    'content': msg.content,
                    'intent': msg.intent
                }
                for msg in history
            ],
            'patient_id': str(request.patient_id),
            'language': language,
            'image_base64': getattr(request, 'image_base64', None),
        }
        
        # Generate response
        result = await response_generator.generate_response(
            request.message,
            conversation_context,
            language
        )
        
        # Add bot message
        bot_message = conversation_manager.add_message(
            conversation.conversation_id,
            'bot',
            result['response'].text,
            language,
            intent=result['intent']['intent'],
            confidence_score=result['intent']['confidence']
        )
        
        # Handle escalation if needed
        escalation_data = result['escalation']
        if escalation_data.get('escalate', False):
            escalation_manager = EscalationManager(db)
            escalation_result = await escalation_manager.escalate_to_counsellor(
                conversation.conversation_id,
                escalation_data,
                request.patient_id
            )
            escalation_data.update(escalation_result)
            conversation_manager.escalate_conversation(conversation.conversation_id)
        
        # Convert intent dict to Intent object
        intent_dict = result['intent']
        intent_obj = Intent(
            name=intent_dict.get('intent', intent_dict.get('name', 'unknown')),
            confidence=intent_dict.get('confidence', 0.0),
            all_probabilities=intent_dict.get('all_probabilities', {}),
            entities=intent_dict.get('entities', [])
        )
        
        return ChatMessageResponse(
            conversation_id=conversation.conversation_id,
            message_id=bot_message.message_id,
            response=result['response'],
            intent=intent_obj,
            escalation=escalation_data,
            timestamp=datetime.utcnow()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.post(
    "/conversation",
    response_model=ConversationResponse,
    summary="Create a new conversation or send message to existing",
    description="""
    Create a new conversation or send a message to an existing conversation.
    
    **New Conversation**: If `conversation_id` is not provided, a new conversation is created.
    If `initial_message` is provided, the bot will respond immediately.
    
    **Existing Conversation**: If `conversation_id` is provided, the message is added to
    that conversation and the bot responds.
    
    **Language**: Can be specified or auto-detected from the message content.
    """,
    response_description="Conversation details with optional bot response",
    responses={
        200: {"description": "Conversation created or message sent successfully"},
        404: {"description": "Conversation not found"},
        403: {"description": "Unauthorized - conversation belongs to different patient"}
    }
)
async def create_conversation(
    request: ConversationRequest,
    db: Session = Depends(get_db),
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
    response_generator: ResponseGenerator = Depends(get_response_generator)
):
    """Start a new conversation or send message to existing conversation"""
    try:
        # Detect language
        language = request.language or settings.DEFAULT_LANGUAGE
        message = request.initial_message  # Can be used for both new and existing conversations
        
        if message:
            detected_lang = language_detector.detect_language(message)
            if language_detector.validate_language(detected_lang):
                language = detected_lang
        
        # If conversation_id is provided, send message to existing conversation
        if request.conversation_id:
            conversation = conversation_manager.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if conversation.patient_id != request.patient_id:
                raise HTTPException(status_code=403, detail="Unauthorized")
            
            # Update language if changed
            if conversation.language != language:
                conversation_manager.update_conversation_language(request.conversation_id, language)
                conversation.language = language
            
            # Add patient message
            if message:
                conversation_manager.add_message(
                    conversation.conversation_id,
                    'patient',
                    message,
                    language
                )
            
            # Get conversation history for context
            history = conversation_manager.get_conversation_history(conversation.conversation_id, limit=10)
            conversation_context = {
                'history': [
                    {
                        'sender': msg.sender_type,
                        'content': msg.content,
                        'intent': msg.intent
                    }
                    for msg in history
                ],
                'patient_id': str(request.patient_id),
                'language': language
            }
            
            # Generate response if message provided
            response = None
            if message:
                result = await response_generator.generate_response(
                    message,
                    conversation_context,
                    language
                )
                response = result['response']
                
                # Add bot message
                conversation_manager.add_message(
                    conversation.conversation_id,
                    'bot',
                    response.text,
                    language,
                    intent=result['intent']['intent'],
                    confidence_score=result['intent']['confidence']
                )
            else:
                # Default greeting if no message
                from app.models.schemas import BotResponse
                templates = response_generator.TEMPLATES_HI if language == 'hi' else response_generator.TEMPLATES_EN
                response = BotResponse(
                    text=templates['greeting'],
                    language=language,
                    suggested_actions=response_generator._get_default_suggestions(language)
                )
            
            return ConversationResponse(
                conversation_id=conversation.conversation_id,
                patient_id=conversation.patient_id,
                status=conversation.status,
                language=conversation.language,
                created_at=conversation.created_at,
                response=response
            )
        
        # Create new conversation
        conversation = conversation_manager.create_conversation(
            request.patient_id,
            language,
            request.metadata
        )
        
        # Generate initial response if message provided
        response = None
        if message:
            conversation_context = {
                'history': [],
                'patient_id': str(request.patient_id),
                'language': language
            }
            result = await response_generator.generate_response(
                message,
                conversation_context,
                language
            )
            response = result['response']
            
            # Add messages
            conversation_manager.add_message(
                conversation.conversation_id,
                'patient',
                message,
                language
            )
            conversation_manager.add_message(
                conversation.conversation_id,
                'bot',
                response.text,
                language
            )
        else:
            # Default greeting
            from app.models.schemas import BotResponse
            templates = response_generator.TEMPLATES_HI if language == 'hi' else response_generator.TEMPLATES_EN
            response = BotResponse(
                text=templates['greeting'],
                language=language,
                suggested_actions=response_generator._get_default_suggestions(language)
            )
        
        return ConversationResponse(
            conversation_id=conversation.conversation_id,
            patient_id=conversation.patient_id,
            status=conversation.status,
            language=conversation.language,
            created_at=conversation.created_at,
            response=response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")


@router.get(
    "/conversation/{conversation_id}",
    response_model=ConversationHistory,
    summary="Get conversation history",
    description="Retrieve the complete message history for a conversation",
    response_description="Conversation history with all messages",
    responses={
        200: {"description": "Conversation history retrieved successfully"},
        404: {"description": "Conversation not found"}
    }
)
async def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Get conversation history"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = conversation_manager.get_conversation_history(conversation_id)
    
    return ConversationHistory(
        conversation_id=conversation.conversation_id,
        patient_id=conversation.patient_id,
        messages=[
            {
                'message_id': str(msg.message_id),
                'sender': msg.sender_type,
                'content': msg.content,
                'language': msg.language,
                'intent': msg.intent,
                'timestamp': msg.created_at.isoformat()
            }
            for msg in messages
        ],
        status=conversation.status,
        language=conversation.language,
        created_at=conversation.created_at,
        last_activity=conversation.last_activity
    )


@router.post(
    "/escalate",
    response_model=EscalationResponse,
    summary="Manually escalate a conversation",
    description="""
    Manually escalate a conversation to a human counsellor.
    
    Use this endpoint when the patient needs human assistance. The urgency level
    determines the priority and estimated wait time.
    
    **Urgency Levels**:
    - `low`: General questions, non-urgent
    - `medium`: Important but not urgent
    - `high`: Needs attention soon
    - `urgent`: Requires immediate attention
    """,
    response_description="Escalation details with estimated wait time",
    responses={
        200: {"description": "Conversation escalated successfully"},
        404: {"description": "Conversation not found"},
        403: {"description": "Unauthorized - conversation belongs to different patient"}
    }
)
async def escalate_conversation(
    request: EscalationRequest,
    db: Session = Depends(get_db),
    escalation_manager: EscalationManager = Depends(get_escalation_manager),
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Manually escalate a conversation"""
    # Verify conversation exists
    conversation = conversation_manager.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.patient_id != request.patient_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    escalation_result = await escalation_manager.escalate_to_counsellor(
        request.conversation_id,
        {
            'reason': request.reason,
            'urgency': request.urgency
        },
        request.patient_id
    )
    
    conversation_manager.escalate_conversation(request.conversation_id)
    
    return EscalationResponse(
        escalation_id=UUID(escalation_result['escalation_id']),
        conversation_id=request.conversation_id,
        status=escalation_result['status'],
        urgency=escalation_result['urgency'],
        estimated_wait_time=escalation_result['estimated_wait_time'],
        message=escalation_result['message']
    )


@router.websocket("/ws/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: UUID):
    """
    WebSocket endpoint for real-time bidirectional chat.
    
    **Connection**: Connect to `ws://localhost:8000/api/v1/chat/ws/{conversation_id}`
    
    **Message Format** (Client → Server):
    ```json
    {
        "content": "Patient message",
        "language": "en"
    }
    ```
    
    **Response Format** (Server → Client):
    ```json
    {
        "type": "response",
        "content": {
            "text": "Bot response",
            "suggested_actions": [...],
            "related_content": [...]
        },
        "intent": {...},
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    
    **Note**: The conversation must exist before connecting via WebSocket.
    """
    await websocket.accept()
    
    try:
        db = next(get_db())
        conversation_manager = ConversationManager(db)
        intent_classifier = IntentClassifier()
        knowledge_engine = KnowledgeEngine(db)
        escalation_manager = EscalationManager(db)
        response_generator = ResponseGenerator(intent_classifier, knowledge_engine, escalation_manager)
        
        # Verify conversation exists
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation:
            await websocket.close(code=1008, reason="Conversation not found")
            return
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message = message_data.get('content', '')
            language = message_data.get('language', conversation.language)
            
            # Add patient message
            conversation_manager.add_message(
                conversation_id,
                'patient',
                message,
                language
            )
            
            # Generate response
            history = conversation_manager.get_conversation_history(conversation_id, limit=10)
            conversation_context = {
                'history': [
                    {
                        'sender': msg.sender_type,
                        'content': msg.content,
                        'intent': msg.intent
                    }
                    for msg in history
                ],
                'patient_id': str(conversation.patient_id),
                'language': language
            }
            
            result = await response_generator.generate_response(
                message,
                conversation_context,
                language
            )
            
            # Add bot message
            conversation_manager.add_message(
                conversation_id,
                'bot',
                result['response'].text,
                language
            )
            
            # Send response
            response_data = {
                'type': 'response',
                'content': result['response'].dict(),
                'intent': result['intent'],
                'timestamp': datetime.utcnow().isoformat()
            }
            await websocket.send_text(json.dumps(response_data))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=1011, reason=str(e))
    finally:
        db.close()
