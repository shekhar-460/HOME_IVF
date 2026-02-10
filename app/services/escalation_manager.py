"""
Escalation Manager - Detect and handle escalations
"""
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from app.database.models import Escalation
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class EscalationManager:
    """Manage escalation to human counsellors"""
    
    # Urgent keywords in English
    URGENT_KEYWORDS_EN = [
        'emergency', 'severe pain', 'bleeding', 'chest pain',
        'difficulty breathing', 'can\'t breathe', 'unconscious',
        'severe', 'critical', 'immediate', 'urgent'
    ]
    
    # Urgent keywords in Hindi
    URGENT_KEYWORDS_HI = [
        'आपातकाल', 'गंभीर दर्द', 'रक्तस्राव', 'सीने में दर्द',
        'सांस लेने में कठिनाई', 'बेहोश', 'गंभीर', 'तत्काल'
    ]
    
    # Emotional distress indicators in English
    EMOTIONAL_INDICATORS_EN = [
        'worried', 'scared', 'anxious', 'depressed', 'hopeless',
        'suicidal', 'overwhelmed', 'can\'t cope', 'breaking down'
    ]
    
    # Emotional distress indicators in Hindi
    EMOTIONAL_INDICATORS_HI = [
        'चिंतित', 'डर', 'चिंता', 'उदास', 'निराश', 'अभिभूत'
    ]
    
    def __init__(self, db: Session):
        self.db = db
    
    async def assess_escalation_need(
        self,
        message: str,
        intent: str,
        conversation_history: List[Dict],
        language: str = "en"
    ) -> Dict:
        """
        Determine if escalation is needed
        
        Args:
            message: User message
            intent: Detected intent
            conversation_history: Previous messages in conversation
            language: Language code
        
        Returns:
            Dict with escalation decision and details
        """
        urgency_score = 0
        complexity_score = 0
        
        message_lower = message.lower()
        
        # Check for urgent keywords
        urgent_keywords = self.URGENT_KEYWORDS_HI if language == 'hi' else self.URGENT_KEYWORDS_EN
        for keyword in urgent_keywords:
            if keyword.lower() in message_lower:
                urgency_score += 2
        
        # Check for emotional distress
        emotional_indicators = self.EMOTIONAL_INDICATORS_HI if language == 'hi' else self.EMOTIONAL_INDICATORS_EN
        for indicator in emotional_indicators:
            if indicator.lower() in message_lower:
                urgency_score += 1
        
        # Check intent (handle None case for parallel processing)
        if intent and intent.startswith('escalation_urgent'):
            urgency_score = 3
        elif intent and intent.startswith('escalation_complex'):
            complexity_score = 2
        
        # Check conversation length (long conversations may need human help)
        if len(conversation_history) > 10:
            complexity_score += 1
        
        # Check for repeated questions (user not getting help)
        if len(conversation_history) > 5:
            recent_intents = [msg.get('intent', '') for msg in conversation_history[-5:]]
            if len(set(recent_intents)) == 1:  # Same intent repeated
                complexity_score += 1
        
        # Decision logic
        if urgency_score >= settings.ESCALATION_URGENCY_THRESHOLD:
            return {
                'escalate': True,
                'urgency': 'urgent',
                'reason': 'urgent_medical_concern',
                'urgency_score': urgency_score,
                'complexity_score': complexity_score
            }
        elif complexity_score >= settings.ESCALATION_COMPLEXITY_THRESHOLD or (intent and intent.startswith('escalation_complex')):
            return {
                'escalate': True,
                'urgency': 'medium',
                'reason': 'complex_query',
                'urgency_score': urgency_score,
                'complexity_score': complexity_score
            }
        else:
            return {
                'escalate': False,
                'urgency': 'low',
                'urgency_score': urgency_score,
                'complexity_score': complexity_score
            }
    
    async def create_escalation(
        self,
        conversation_id: UUID,
        reason: str,
        urgency: str,
        patient_id: UUID
    ) -> Escalation:
        """Create escalation record"""
        escalation = Escalation(
            escalation_id=uuid4(),
            conversation_id=conversation_id,
            reason=reason,
            urgency=urgency,
            status='pending'
        )
        self.db.add(escalation)
        self.db.commit()
        self.db.refresh(escalation)
        
        logger.info(f"Created escalation {escalation.escalation_id} for conversation {conversation_id}")
        
        # TODO: Send notification to counsellor system
        # await self._notify_counsellors(escalation)
        
        return escalation
    
    def _estimate_wait_time(self, urgency: str) -> int:
        """Estimate wait time in minutes based on urgency"""
        wait_times = {
            'urgent': 2,
            'high': 5,
            'medium': 15,
            'low': 30
        }
        return wait_times.get(urgency, 15)
    
    async def escalate_to_counsellor(
        self,
        conversation_id: UUID,
        escalation_data: Dict,
        patient_id: UUID
    ) -> Dict:
        """Route conversation to human counsellor"""
        escalation = await self.create_escalation(
            conversation_id=conversation_id,
            reason=escalation_data.get('reason', 'complex_query'),
            urgency=escalation_data.get('urgency', 'medium'),
            patient_id=patient_id
        )
        
        wait_time = self._estimate_wait_time(escalation_data.get('urgency', 'medium'))
        
        return {
            'escalation_id': str(escalation.escalation_id),
            'status': escalation.status,
            'urgency': escalation.urgency,
            'estimated_wait_time': wait_time,
            'message': f"I'm connecting you with a counsellor. Estimated wait time: {wait_time} minutes."
        }
