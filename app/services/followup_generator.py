"""
Follow-up Question Generator - Generate contextual follow-up questions
"""
from typing import List, Dict, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class FollowupGenerator:
    """Generate contextual follow-up questions based on topics and answers"""
    
    # Follow-up question templates by category (English)
    FOLLOWUP_TEMPLATES_EN = {
        'ivf_process': [
            "How long does the IVF process take?",
            "What are the success rates of IVF?",
            "What happens during egg retrieval?",
            "What should I expect after embryo transfer?"
        ],
        'medication': [
            "What are the common side effects of IVF medications?",
            "How do I take IVF injections?",
            "What should I do if I miss a dose?",
            "Are there any dietary restrictions with these medications?",
            "How long do I need to take these medications?"
        ],
        'side_effects': [
            "When should I contact my doctor during IVF?",
            "What are normal side effects vs. concerning symptoms?",
            "How can I manage side effects?",
            "What medications can help with side effects?"
        ],
        'lifestyle': [
            "What should I eat during IVF treatment?",
            "Can I exercise during IVF?",
            "Are there activities I should avoid?",
            "How should I manage stress during treatment?"
        ],
        'success_factors': [
            "What factors affect IVF success rates?",
            "How can I improve my chances of success?",
            "What is the average number of cycles needed?",
            "What happens if the first cycle doesn't work?"
        ],
        'costs': [
            "Does insurance cover IVF?",
            "Are there financing options available?",
            "What is included in the IVF cost?",
            "Are there any hidden costs?"
        ],
        'general': [
            "What is the first step in starting IVF?",
            "How do I prepare for IVF treatment?",
            "What questions should I ask my doctor?",
            "What support resources are available?"
        ]
    }
    
    # Follow-up question templates by category (Hindi)
    FOLLOWUP_TEMPLATES_HI = {
        'ivf_process': [
            "आईवीएफ प्रक्रिया में कितना समय लगता है?",
            "आईवीएफ की सफलता दर क्या है?",
            "अंडा पुनर्प्राप्ति के दौरान क्या होता है?",
            "भ्रूण स्थानांतरण के बाद मुझे क्या उम्मीद करनी चाहिए?"
        ],
        'medication': [
            "आईवीएफ दवाओं के सामान्य दुष्प्रभाव क्या हैं?",
            "मैं आईवीएफ इंजेक्शन कैसे लूं?",
            "अगर मैं एक खुराक भूल जाऊं तो क्या करूं?",
            "इन दवाओं के साथ कोई आहार प्रतिबंध हैं?",
            "मुझे इन दवाओं को कितने समय तक लेना होगा?"
        ],
        'side_effects': [
            "आईवीएफ के दौरान मुझे अपने डॉक्टर से कब संपर्क करना चाहिए?",
            "सामान्य दुष्प्रभाव बनाम चिंताजनक लक्षण क्या हैं?",
            "मैं दुष्प्रभावों का प्रबंधन कैसे कर सकता हूं?",
            "कौन सी दवाएं दुष्प्रभावों में मदद कर सकती हैं?"
        ],
        'lifestyle': [
            "आईवीएफ उपचार के दौरान मुझे क्या खाना चाहिए?",
            "क्या मैं आईवीएफ के दौरान व्यायाम कर सकती हूं?",
            "क्या कोई गतिविधियां हैं जिनसे मुझे बचना चाहिए?",
            "उपचार के दौरान मैं तनाव का प्रबंधन कैसे करूं?"
        ],
        'success_factors': [
            "कौन से कारक आईवीएफ सफलता दर को प्रभावित करते हैं?",
            "मैं अपनी सफलता की संभावनाओं को कैसे बेहतर बना सकता हूं?",
            "आवश्यक चक्रों की औसत संख्या क्या है?",
            "अगर पहला चक्र काम नहीं करता तो क्या होता है?"
        ],
        'costs': [
            "क्या बीमा आईवीएफ को कवर करता है?",
            "क्या वित्तपोषण विकल्प उपलब्ध हैं?",
            "आईवीएफ लागत में क्या शामिल है?",
            "क्या कोई छुपी हुई लागतें हैं?"
        ],
        'general': [
            "आईवीएफ शुरू करने में पहला कदम क्या है?",
            "मैं आईवीएफ उपचार के लिए कैसे तैयारी करूं?",
            "मुझे अपने डॉक्टर से कौन से प्रश्न पूछने चाहिए?",
            "कौन से सहायता संसाधन उपलब्ध हैं?"
        ]
    }
    
    def __init__(self):
        self.max_followups = settings.MAX_FOLLOWUP_QUESTIONS
    
    def generate_followups(
        self,
        category: str,
        language: str = "en",
        answer: Optional[str] = None,
        exclude_questions: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate contextual follow-up questions
        
        Args:
            category: FAQ category (e.g., 'ivf_process', 'medication')
            language: Language code ('en' or 'hi')
            answer: The answer that was provided (for context)
            exclude_questions: Questions to exclude from results
        
        Returns:
            List of follow-up questions
        """
        if not settings.ENABLE_FOLLOWUPS:
            return []
        
        # Medication is a specialised opinion - do not suggest medication follow-ups
        if category == 'medication':
            return []
        
        # Get templates based on language
        templates = self.FOLLOWUP_TEMPLATES_HI if language == 'hi' else self.FOLLOWUP_TEMPLATES_EN
        
        # Get questions for the category, fallback to 'general'
        questions = templates.get(category, templates.get('general', []))
        
        # Exclude questions if provided
        if exclude_questions:
            questions = [q for q in questions if q not in exclude_questions]
        
        # Return top N questions
        return questions[:self.max_followups]
    
    def get_followups_for_intent(
        self,
        intent: str,
        language: str = "en"
    ) -> List[str]:
        """
        Get follow-up questions based on intent
        
        Args:
            intent: Detected intent (e.g., 'faq_process', 'faq_medication')
            language: Language code
        
        Returns:
            List of follow-up questions
        """
        # Extract category from intent (e.g., 'faq_process' -> 'process')
        if intent.startswith('faq_'):
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
        
        return self.generate_followups(category, language)
    
    def generate_contextual_followups(
        self,
        query: str,
        answer: str,
        category: str,
        language: str = "en"
    ) -> List[str]:
        """
        Generate follow-up questions based on the actual query and answer
        
        Args:
            query: Original user query
            answer: Provided answer
            category: FAQ category
            language: Language code
        
        Returns:
            List of contextual follow-up questions
        """
        # Start with template-based questions
        followups = self.generate_followups(category, language, answer)
        
        # Could enhance with AI-based generation in the future
        # For now, return template-based questions
        return followups
