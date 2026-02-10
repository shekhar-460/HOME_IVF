"""
Admin endpoints for managing knowledge base and analytics
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from datetime import datetime, timedelta
import json

from app.database.connection import get_db
from app.database.models import FAQ, Article, Conversation, Message, Escalation
from app.services.knowledge_engine import KnowledgeEngine, MedgemmaModelManager

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/analytics",
    summary="Get bot analytics",
    description="""
    Retrieve analytics and statistics about bot usage.
    
    Returns metrics including:
    - Total conversations and messages
    - FAQ resolution rate
    - Escalation rate
    - Top intents
    - Language distribution
    
    **Time Range**: Specify number of days to analyze (default: 30 days).
    """,
    response_description="Analytics data with various metrics",
    tags=["admin"]
)
async def get_analytics(
    db: Session = Depends(get_db),
    days: int = 30
):
    """Get bot analytics"""
    try:
        # Date range
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Total conversations
        total_conversations = db.query(Conversation).filter(
            Conversation.created_at >= start_date
        ).count()
        
        # Total messages
        total_messages = db.query(Message).filter(
            Message.created_at >= start_date
        ).count()
        
        # FAQ resolution rate (messages with high confidence)
        resolved_messages = db.query(Message).filter(
            Message.created_at >= start_date,
            Message.sender_type == 'bot',
            Message.confidence_score >= 0.7
        ).count()
        faq_resolution_rate = resolved_messages / total_messages if total_messages > 0 else 0
        
        # Escalation rate
        total_escalations = db.query(Escalation).filter(
            Escalation.created_at >= start_date
        ).count()
        escalation_rate = total_escalations / total_conversations if total_conversations > 0 else 0
        
        # Top intents
        top_intents = db.query(
            Message.intent,
            db.func.count(Message.message_id).label('count')
        ).filter(
            Message.created_at >= start_date,
            Message.intent.isnot(None)
        ).group_by(Message.intent).order_by(db.func.count(Message.message_id).desc()).limit(10).all()
        
        # Language distribution
        language_dist = db.query(
            Conversation.language,
            db.func.count(Conversation.conversation_id).label('count')
        ).filter(
            Conversation.created_at >= start_date
        ).group_by(Conversation.language).all()
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "faq_resolution_rate": round(faq_resolution_rate, 2),
            "escalation_rate": round(escalation_rate, 2),
            "top_intents": [{"intent": intent, "count": count} for intent, count in top_intents],
            "language_distribution": [{"language": lang, "count": count} for lang, count in language_dist],
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")


@router.post(
    "/faq",
    summary="Add new FAQ",
    description="""
    Add a new FAQ to the knowledge base.
    
    The FAQ will be automatically:
    - Indexed for semantic search
    - Available in both languages (if provided)
    - Categorized for better organization
    
    **Categories**: ivf_process, medication, side_effects, lifestyle, success_factors, costs, general
    """,
    response_description="Created FAQ details",
    tags=["admin"]
)
async def create_faq(
    question: str,
    answer: str,
    question_hi: str = None,
    answer_hi: str = None,
    category: str = None,
    tags: List[str] = [],
    db: Session = Depends(get_db)
):
    """Add new FAQ"""
    try:
        # Generate embedding
        knowledge_engine = KnowledgeEngine(db)
        content = f"{question} {answer}"
        embedding = knowledge_engine._get_embedding(content)
        
        faq = FAQ(
            question=question,
            answer=answer,
            question_hi=question_hi,
            answer_hi=answer_hi,
            category=category,
            tags=tags,
            embedding=json.dumps(embedding) if embedding else None
        )
        
        db.add(faq)
        db.commit()
        db.refresh(faq)
        
        return {
            "faq_id": str(faq.faq_id),
            "question": faq.question,
            "answer": faq.answer,
            "status": "created"
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating FAQ: {str(e)}")


@router.put(
    "/faq/{faq_id}",
    summary="Update FAQ",
    description="Update an existing FAQ in the knowledge base. Only provided fields will be updated.",
    response_description="Updated FAQ details",
    tags=["admin"],
    responses={
        200: {"description": "FAQ updated successfully"},
        404: {"description": "FAQ not found"}
    }
)
async def update_faq(
    faq_id: UUID,
    question: str = None,
    answer: str = None,
    question_hi: str = None,
    answer_hi: str = None,
    category: str = None,
    tags: List[str] = None,
    db: Session = Depends(get_db)
):
    """Update FAQ"""
    faq = db.query(FAQ).filter(FAQ.faq_id == faq_id).first()
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    try:
        if question is not None:
            faq.question = question
        if answer is not None:
            faq.answer = answer
        if question_hi is not None:
            faq.question_hi = question_hi
        if answer_hi is not None:
            faq.answer_hi = answer_hi
        if category is not None:
            faq.category = category
        if tags is not None:
            faq.tags = tags
        
        # Regenerate embedding if content changed
        if question or answer:
            knowledge_engine = KnowledgeEngine(db)
            content = f"{faq.question} {faq.answer}"
            embedding = knowledge_engine._get_embedding(content)
            faq.embedding = json.dumps(embedding) if embedding else faq.embedding
        
        db.commit()
        db.refresh(faq)
        
        return {
            "faq_id": str(faq.faq_id),
            "status": "updated"
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating FAQ: {str(e)}")


@router.post(
    "/article",
    summary="Add educational article",
    description="""
    Add a new educational article to the knowledge base.
    
    Articles provide detailed information about IVF topics and are searchable
    through semantic search. Articles can be longer than FAQs and provide
    comprehensive information.
    """,
    response_description="Created article details",
    tags=["admin"]
)
async def create_article(
    title: str,
    content: str,
    title_hi: str = None,
    content_hi: str = None,
    category: str = None,
    tags: List[str] = [],
    db: Session = Depends(get_db)
):
    """Add educational article"""
    try:
        # Generate embedding
        knowledge_engine = KnowledgeEngine(db)
        full_content = f"{title} {content}"
        embedding = knowledge_engine._get_embedding(full_content)
        
        article = Article(
            title=title,
            content=content,
            title_hi=title_hi,
            content_hi=content_hi,
            category=category,
            tags=tags,
            embedding=json.dumps(embedding) if embedding else None
        )
        
        db.add(article)
        db.commit()
        db.refresh(article)
        
        return {
            "article_id": str(article.article_id),
            "title": article.title,
            "status": "created"
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating article: {str(e)}")


@router.post(
    "/gpu/cleanup",
    summary="Force GPU memory cleanup",
    description="""
    Manually trigger GPU memory cleanup for the Medgemma model.
    
    This endpoint forces the release of GPU memory by unloading the Medgemma model.
    The model will be reloaded automatically on the next request that needs it.
    
    **Use Cases**:
    - Free up GPU memory when not actively using the bot
    - Troubleshoot GPU memory issues
    - Manual memory management
    
    **Note**: This will temporarily slow down the next request that uses Medgemma,
    as the model will need to be reloaded.
    """,
    response_description="Status of GPU memory cleanup",
    tags=["admin"]
)
async def cleanup_gpu_memory():
    """Force cleanup of GPU memory"""
    try:
        model_manager = MedgemmaModelManager()
        model_manager.cleanup(force=True)
        
        # Get GPU memory info if available
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    gpu_info[f"gpu_{i}"] = {
                        "allocated_gb": round(allocated, 2),
                        "reserved_gb": round(reserved, 2)
                    }
        except:
            pass
        
        return {
            "status": "success",
            "message": "GPU memory cleanup completed",
            "gpu_info": gpu_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up GPU memory: {str(e)}")
