"""
Proactive Suggestions Engine - Generate proactive topic suggestions
"""
from typing import List, Dict, Optional
from app.models.schemas import SuggestedAction
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ProactiveSuggestions:
    """Generate proactive topic suggestions based on conversation context"""
    
    # Topic mapping - related topics for each category (medication excluded - specialised opinion)
    TOPIC_MAPPING = {
        'ivf_process': ['side_effects', 'success_factors', 'lifestyle'],
        'medication': ['side_effects', 'lifestyle', 'ivf_process'],
        'side_effects': ['when_to_call', 'lifestyle'],
        'lifestyle': ['ivf_process', 'success_factors'],
        'success_factors': ['ivf_process', 'lifestyle', 'costs'],
        'costs': ['ivf_process', 'success_factors'],
        'general': ['ivf_process', 'lifestyle']
    }
    
    # Topic labels in English
    TOPIC_LABELS_EN = {
        'ivf_process': 'IVF Process',
        'medication': 'Medications',
        'side_effects': 'Side Effects',
        'lifestyle': 'Lifestyle & Diet',
        'success_factors': 'Success Rates',
        'costs': 'Costs & Insurance',
        'when_to_call': 'When to Call Doctor',
        'general': 'General Information'
    }
    
    # Topic labels in Hindi
    TOPIC_LABELS_HI = {
        'ivf_process': 'आईवीएफ प्रक्रिया',
        'medication': 'दवाएं',
        'side_effects': 'दुष्प्रभाव',
        'lifestyle': 'जीवनशैली और आहार',
        'success_factors': 'सफलता दर',
        'costs': 'लागत और बीमा',
        'when_to_call': 'डॉक्टर को कब बुलाएं',
        'general': 'सामान्य जानकारी'
    }
    
    # Quick action suggestions in English
    QUICK_ACTIONS_EN = {
        'schedule_appointment': 'Schedule Appointment',
        'view_faqs': 'View All FAQs',
        'contact_support': 'Contact Support',
        'view_resources': 'View Resources'
    }
    
    # Quick action suggestions in Hindi
    QUICK_ACTIONS_HI = {
        'schedule_appointment': 'अपॉइंटमेंट शेड्यूल करें',
        'view_faqs': 'सभी FAQs देखें',
        'contact_support': 'सहायता से संपर्क करें',
        'view_resources': 'संसाधन देखें'
    }
    
    def __init__(self):
        pass
    
    def get_suggestions(
        self,
        current_topic: str,
        language: str = "en",
        conversation_history: Optional[List[Dict]] = None
    ) -> List[SuggestedAction]:
        """
        Get proactive suggestions based on current topic
        
        Args:
            current_topic: Current topic/category
            language: Language code ('en' or 'hi')
            conversation_history: Optional conversation history for context
        
        Returns:
            List of suggested actions
        """
        if not settings.ENABLE_PROACTIVE_SUGGESTIONS:
            return []
        
        suggestions = []
        
        # Ensure current_topic is not None or empty
        if not current_topic:
            current_topic = 'general'
        
        # Get related topics
        related_topics = self.TOPIC_MAPPING.get(current_topic, [])
        
        # Medication is a specialised opinion - never suggest it in AI response
        related_topics = [t for t in related_topics if t != 'medication']
        
        # Get labels based on language
        topic_labels = self.TOPIC_LABELS_HI if language == 'hi' else self.TOPIC_LABELS_EN
        
        # Filter out topics already discussed
        discussed_topics = set()
        if conversation_history:
            for msg in conversation_history[-10:]:  # Check last 10 messages
                intent = msg.get('intent') or ''
                if intent and intent.startswith('faq_'):
                    topic = intent.replace('faq_', '')
                    discussed_topics.add(topic)
        
        # Create suggestions for related topics (max 3)
        for topic in related_topics[:3]:
            if topic not in discussed_topics and topic != 'medication':
                label = topic_labels.get(topic, topic)
                suggestions.append(SuggestedAction(
                    type='quick_reply',
                    label=label,
                    action=f'faq_{topic}',
                    data={'category': topic}
                ))
        
        return suggestions
    
    def get_suggestions_for_intent(
        self,
        intent: str,
        language: str = "en",
        conversation_history: Optional[List[Dict]] = None
    ) -> List[SuggestedAction]:
        """
        Get suggestions based on intent
        
        Args:
            intent: Detected intent
            language: Language code
            conversation_history: Optional conversation history
        
        Returns:
            List of suggested actions
        """
        # Extract category from intent
        if intent and intent.startswith('faq_'):
            category = intent.replace('faq_', '')
            # Map to actual category names
            category_mapping = {
                'process': 'ivf_process',
                'medication': 'medication',
                'side_effects': 'side_effects',
                'lifestyle': 'lifestyle',
                'success_factors': 'success_factors',
                'costs': 'costs',
                'general': 'general'
            }
            category = category_mapping.get(category, 'general')
        else:
            category = 'general'
        
        return self.get_suggestions(category, language, conversation_history)
    
    def get_quick_actions(
        self,
        language: str = "en"
    ) -> List[SuggestedAction]:
        """
        Get quick action suggestions
        
        Args:
            language: Language code
        
        Returns:
            List of quick action suggestions
        """
        actions = self.QUICK_ACTIONS_HI if language == 'hi' else self.QUICK_ACTIONS_EN
        
        return [
            SuggestedAction(
                type='button',
                label=actions['schedule_appointment'],
                action='appointment_schedule'
            ),
            SuggestedAction(
                type='button',
                label=actions['view_faqs'],
                action='view_faqs'
            ),
            SuggestedAction(
                type='button',
                label=actions['contact_support'],
                action='contact_support'
            )
        ]
    
    def analyze_conversation(
        self,
        conversation_history: List[Dict]
    ) -> Dict:
        """
        Analyze conversation to identify topics and concerns
        
        Args:
            conversation_history: List of conversation messages
        
        Returns:
            Dict with analysis results
        """
        discussed_topics = set()
        concerns = []
        
        for msg in conversation_history:
            intent = msg.get('intent') or ''
            content = msg.get('content') or ''
            
            # Track discussed topics
            if intent and intent.startswith('faq_'):
                topic = intent.replace('faq_', '')
                discussed_topics.add(topic)
            
            # Identify concerns (simple keyword-based)
            concern_keywords = ['worried', 'concerned', 'afraid', 'anxious', 'scared']
            if any(keyword in content.lower() for keyword in concern_keywords):
                concerns.append(content[:100])  # First 100 chars
        
        return {
            'discussed_topics': list(discussed_topics),
            'concerns': concerns,
            'message_count': len(conversation_history)
        }
