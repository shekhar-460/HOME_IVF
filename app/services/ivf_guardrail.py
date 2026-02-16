"""
IVF Guardrail Service - Ensures all responses are strictly related to IVF only
"""
from typing import Dict, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class IVFGuardrail:
    """Guardrail to restrict bot to IVF-related information only"""
    
    # IVF-related keywords (English)
    IVF_KEYWORDS_EN = [
        # Core IVF terms
        'ivf', 'in vitro fertilization', 'in-vitro', 'test tube baby',
        'fertility', 'fertility treatment', 'fertility clinic',
        'infertility', 'infertile', 'conception', 'pregnancy',
        'embryo', 'embryos', 'embryo transfer', 'embryo implantation',
        'egg retrieval', 'egg collection', 'oocyte', 'oocytes',
        'sperm', 'sperm collection', 'sperm analysis',
        'fertilization', 'fertilized egg',
        'implantation', 'implant', 'implanted',
        'blastocyst', 'blastocysts',
        
        # IVF process terms
        'ivf cycle', 'ivf treatment', 'ivf procedure', 'ivf process',
        'stimulation', 'ovarian stimulation', 'ovulation',
        'follicle', 'follicles', 'follicular',
        'trigger shot', 'hcg injection', 'lupron', 'gonal', 'menopur',
        'retrieval', 'egg retrieval', 'oocyte retrieval',
        'transfer', 'embryo transfer', 'frozen transfer', 'fresh transfer',
        'frozen embryo', 'frozen embryos', 'embryo freezing',
        
        # Medications
        'gonadotropin', 'gonadotropins', 'fsh', 'lh',
        'clomid', 'letrozole', 'metformin',
        'progesterone', 'estrogen', 'estradiol',
        'cetrotide', 'ganirelix', 'antagonist',
        'ovidrel', 'pregnyl', 'trigger',
        
        # IVF-related conditions and topics
        'pcos', 'polycystic ovary', 'endometriosis',
        'male infertility', 'low sperm count', 'azoospermia',
        'tubal factor', 'blocked tubes',
        'age and ivf', 'ivf success rate', 'ivf success',
        'pregnancy rate', 'live birth rate',
        
        # Side effects and symptoms
        'ohss', 'ovarian hyperstimulation', 'bloating', 'cramping',
        'injection site', 'side effect', 'side effects',
        
        # Lifestyle and preparation
        'ivf diet', 'ivf lifestyle', 'ivf preparation',
        'before ivf', 'during ivf', 'after ivf',
        'ivf cost', 'ivf price', 'ivf expenses',
        
        # Related procedures
        'icsi', 'intracytoplasmic', 'pgd', 'pgs', 'genetic testing',
        'iui', 'intrauterine insemination',
        'surrogacy', 'gestational carrier',
        'egg donor', 'sperm donor', 'donor egg', 'donor sperm',
        
        # Hindi transliterations and common terms
        'test tube baby', 'ivf baby',
    ]
    
    # IVF-related keywords (Hindi - transliterated and Devanagari)
    # Include Latin 'ivf' so mixed-script queries like "ivf kya hai?" are recognized
    IVF_KEYWORDS_HI = [
        # Core terms in Hindi (and Latin for Hinglish e.g. "ivf kya hai?")
        'ivf', 'आईवीएफ', 'इन विट्रो', 'टेस्ट ट्यूब बेबी',
        'गर्भधारण', 'गर्भावस्था', 'बांझपन', 'बांझ',
        'भ्रूण', 'भ्रूण स्थानांतरण', 'भ्रूण प्रत्यारोपण',
        'अंडा', 'अंडे', 'अंडा संग्रह', 'अंडाशय',
        'शुक्राणु', 'शुक्राणु संग्रह', 'शुक्राणु विश्लेषण',
        'निषेचन', 'निषेचित अंडा',
        'प्रत्यारोपण', 'प्रत्यारोपित',
        
        # Process terms
        'आईवीएफ चक्र', 'आईवीएफ उपचार', 'आईवीएफ प्रक्रिया',
        'उत्तेजना', 'अंडाशय उत्तेजना', 'ओव्यूलेशन',
        'कूप', 'कूपिक',
        'ट्रिगर शॉट', 'एचसीजी इंजेक्शन',
        'स्थानांतरण', 'भ्रूण स्थानांतरण',
        'जमे हुए भ्रूण', 'ताजा स्थानांतरण',
        
        # Medications
        'दवा', 'दवाएं', 'इंजेक्शन', 'खुराक',
        'प्रोजेस्टेरोन', 'एस्ट्रोजन',
        
        # Conditions
        'पीसीओएस', 'एंडोमेट्रियोसिस',
        'पुरुष बांझपन', 'कम शुक्राणु',
        
        # Success and rates
        'सफलता दर', 'गर्भावस्था दर',
        
        # Side effects
        'दुष्प्रभाव', 'समस्या', 'दर्द',
        
        # Lifestyle
        'आहार', 'जीवनशैली', 'तैयारी',
        'लागत', 'मूल्य', 'खर्च',
    ]
    
    # Non-IVF topics to detect and reject
    NON_IVF_TOPICS_EN = [
        # General health (not IVF-specific)
        r'\b(cancer|tumor|carcinoma|chemotherapy|radiation)\b',
        r'\b(heart attack|stroke|diabetes|hypertension|blood pressure)\b',
        r'\b(covid|coronavirus|pandemic|vaccine)\b',
        r'\b(cold|flu|fever|cough|sore throat)\b',
        r'\b(headache|migraine|back pain|joint pain)\b',
        r'\b(depression|anxiety|mental health|psychology)\b',
        r'\b(diet|weight loss|obesity|exercise|fitness)\b',  # Only if not IVF-related
        r'\b(surgery|operation|procedure)\b',  # Only if not IVF-related
        
        # Other medical specialties
        r'\b(dermatology|skin|acne|rash)\b',
        r'\b(orthopedic|bone|fracture|broken)\b',
        r'\b(neurology|brain|seizure|epilepsy)\b',
        r'\b(cardiology|heart|cardiac)\b',
        r'\b(oncology|tumor|cancer treatment)\b',
        
        # Non-medical topics
        r'\b(weather|news|sports|entertainment|movie|music)\b',
        r'\b(cooking|recipe|food recipe)\b',
        r'\b(travel|vacation|hotel|flight)\b',
        r'\b(shopping|buy|purchase|price of)\b',  # Only if not IVF cost
    ]
    
    # Non-IVF topics in Hindi
    NON_IVF_TOPICS_HI = [
        r'\b(कैंसर|ट्यूमर|कीमोथेरेपी)\b',
        r'\b(दिल का दौरा|स्ट्रोक|मधुमेह|रक्तचाप)\b',
        r'\b(कोविड|कोरोनावायरस|वैक्सीन)\b',
        r'\b(सर्दी|जुकाम|बुखार|खांसी)\b',
        r'\b(सिरदर्द|माइग्रेन|पीठ दर्द)\b',
        r'\b(अवसाद|चिंता|मानसिक स्वास्थ्य)\b',
        r'\b(त्वचा|एक्ने|दाने)\b',
        r'\b(हड्डी|फ्रैक्चर)\b',
        r'\b(मौसम|समाचार|खेल|मनोरंजन)\b',
    ]
    
    # Response templates for non-IVF queries
    REJECTION_TEMPLATES_EN = {
        'not_ivf_related': (
            "I'm specialized in providing information about IVF (In Vitro Fertilization) treatment only. "
            "I can help you with questions about:\n\n"
            "• IVF procedures and processes\n"
            "• IVF medications and injections\n"
            "• IVF success rates and factors\n"
            "• Side effects during IVF treatment\n"
            "• Lifestyle and preparation for IVF\n"
            "• IVF costs and financial aspects\n"
            "• Related fertility treatments (ICSI, IUI, etc.)\n\n"
            "If you have questions about IVF, I'm here to help! What would you like to know?"
        ),
        'unclear': (
            "I'm here to help with IVF-related questions. Could you please clarify how your question relates to IVF treatment? "
            "If you have questions about IVF procedures, medications, success rates, or any other IVF-related topics, I'd be happy to help!"
        )
    }
    
    REJECTION_TEMPLATES_HI = {
        'not_ivf_related': (
            "मैं केवल आईवीएफ (इन विट्रो फर्टिलाइजेशन) उपचार के बारे में जानकारी प्रदान करने में विशेषज्ञ हूं। "
            "मैं आपकी इन विषयों पर मदद कर सकता हूं:\n\n"
            "• आईवीएफ प्रक्रियाएं और प्रक्रिया\n"
            "• आईवीएफ दवाएं और इंजेक्शन\n"
            "• आईवीएफ सफलता दर और कारक\n"
            "• आईवीएफ उपचार के दौरान दुष्प्रभाव\n"
            "• आईवीएफ के लिए जीवनशैली और तैयारी\n"
            "• आईवीएफ लागत और वित्तीय पहलू\n"
            "• संबंधित प्रजनन उपचार (ICSI, IUI, आदि)\n\n"
            "यदि आपके पास आईवीएफ के बारे में प्रश्न हैं, तो मैं मदद के लिए यहां हूं! आप क्या जानना चाहेंगे?"
        ),
        'unclear': (
            "मैं आईवीएफ-संबंधित प्रश्नों में मदद करने के लिए यहां हूं। क्या आप कृपया स्पष्ट कर सकते हैं कि आपका प्रश्न आईवीएफ उपचार से कैसे संबंधित है? "
            "यदि आपके पास आईवीएफ प्रक्रियाओं, दवाओं, सफलता दरों, या किसी अन्य आईवीएफ-संबंधित विषयों के बारे में प्रश्न हैं, तो मैं मदद करने में खुश हूंगा!"
        )
    }
    
    def __init__(self):
        """Initialize IVF guardrail"""
        # Compile regex patterns for non-IVF topics
        self.non_ivf_patterns_en = [re.compile(pattern, re.IGNORECASE) for pattern in self.NON_IVF_TOPICS_EN]
        self.non_ivf_patterns_hi = [re.compile(pattern, re.IGNORECASE) for pattern in self.NON_IVF_TOPICS_HI]
        
        # Create keyword sets for faster lookup
        self.ivf_keywords_en_set = set(kw.lower() for kw in self.IVF_KEYWORDS_EN)
        self.ivf_keywords_hi_set = set(kw.lower() for kw in self.IVF_KEYWORDS_HI)
    
    def is_ivf_related(
        self,
        query: str,
        language: str = "en",
        context: Optional[Dict] = None
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check if query is related to IVF
        
        Args:
            query: User query
            language: Language code ('en' or 'hi')
            context: Optional conversation context
        
        Returns:
            Tuple of (is_ivf_related, confidence_score, rejection_reason)
            - is_ivf_related: True if query is IVF-related
            - confidence_score: Confidence score (0.0 to 1.0)
            - rejection_reason: Reason for rejection if not IVF-related
        """
        query_lower = query.lower().strip()
        
        # Check for non-IVF topics first (strict rejection)
        non_ivf_patterns = self.non_ivf_patterns_hi if language == 'hi' else self.non_ivf_patterns_en
        
        for pattern in non_ivf_patterns:
            if pattern.search(query_lower):
                # Check if it's actually IVF-related despite matching pattern
                # (e.g., "IVF cost" might match "price" pattern but is IVF-related)
                if not self._has_ivf_context(query_lower, language):
                    logger.debug(f"Non-IVF topic detected: {pattern.pattern}")
                    return False, 0.0, 'not_ivf_related'
        
        # Check for IVF keywords
        ivf_keywords = self.ivf_keywords_hi_set if language == 'hi' else self.ivf_keywords_en_set
        
        # Count IVF keyword matches
        matches = sum(1 for keyword in ivf_keywords if keyword in query_lower)
        
        # Check conversation context for IVF-related history
        context_score = 0.0
        if context and context.get('history'):
            recent_messages = context['history'][-3:]  # Last 3 messages
            context_ivf_matches = sum(
                1 for msg in recent_messages
                if any(kw in msg.get('content', '').lower() for kw in ivf_keywords)
            )
            context_score = min(context_ivf_matches * 0.2, 0.4)  # Max 0.4 from context
        
        # Calculate confidence score
        keyword_score = min(matches * 0.3, 0.6)  # Max 0.6 from keywords
        total_score = keyword_score + context_score
        
        # Threshold: need at least some IVF-related content
        if matches > 0 or total_score >= 0.3:
            return True, min(total_score, 1.0), None
        
        # If no clear IVF connection, check if it's a greeting or appointment
        if self._is_greeting_or_appointment(query_lower, language):
            return True, 0.5, None  # Allow greetings and appointments
        
        # Unclear - might be IVF-related but not obvious
        if len(query_lower.split()) <= 3:  # Very short queries might be unclear
            return False, 0.2, 'unclear'
        
        # Not IVF-related
        return False, 0.0, 'not_ivf_related'
    
    def _has_ivf_context(self, query_lower: str, language: str) -> bool:
        """Check if query has IVF context despite matching non-IVF pattern"""
        ivf_keywords = self.ivf_keywords_hi_set if language == 'hi' else self.ivf_keywords_en_set
        return any(kw in query_lower for kw in ivf_keywords)
    
    def _is_greeting_or_appointment(self, query_lower: str, language: str) -> bool:
        """Check if query is a greeting or appointment request"""
        if language == 'hi':
            greetings = ['नमस्ते', 'हैलो', 'नमस्कार', 'धन्यवाद', 'शुक्रिया']
            appointments = ['अपॉइंटमेंट', 'समय', 'स्लॉट', 'बुक']
        else:
            greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'thanks', 'thank you']
            appointments = ['appointment', 'schedule', 'book', 'slot', 'available']
        
        all_terms = greetings + appointments
        return any(term in query_lower for term in all_terms)
    
    def get_rejection_message(self, reason: str, language: str = "en") -> str:
        """Get rejection message for non-IVF queries"""
        templates = self.REJECTION_TEMPLATES_HI if language == 'hi' else self.REJECTION_TEMPLATES_EN
        return templates.get(reason, templates.get('not_ivf_related', ''))
    
    def validate_response(self, response_text: str, original_query: str, language: str = "en") -> Tuple[bool, Optional[str]]:
        """
        Validate that generated response is IVF-related
        
        Args:
            response_text: Generated response text
            original_query: Original user query
            language: Language code
        
        Returns:
            Tuple of (is_valid, rejection_message)
        """
        # Check if response mentions IVF-related terms
        response_lower = response_text.lower()
        ivf_keywords = self.ivf_keywords_hi_set if language == 'hi' else self.ivf_keywords_en_set
        
        # If response has IVF keywords, it's likely valid
        if any(kw in response_lower for kw in ivf_keywords):
            return True, None
        
        # If original query was IVF-related, allow response even if it doesn't explicitly mention IVF
        # (e.g., "How long?" in context of IVF cycle)
        is_query_ivf, _, _ = self.is_ivf_related(original_query, language)
        if is_query_ivf:
            return True, None
        
        # Response doesn't seem IVF-related
        logger.warning(f"Response validation failed: response doesn't appear IVF-related")
        return False, self.get_rejection_message('not_ivf_related', language)
    
    def enhance_ivf_context(self, query: str, language: str = "en") -> str:
        """
        Enhance query with IVF context for better AI model understanding
        
        Args:
            query: Original query
            language: Language code
        
        Returns:
            Enhanced query with IVF context
        """
        # Check if query already has IVF context
        is_ivf, confidence, _ = self.is_ivf_related(query, language)
        
        if is_ivf and confidence >= 0.5:
            # Query is clearly IVF-related, add explicit context
            if language == 'hi':
                return f"आईवीएफ उपचार के संदर्भ में: {query}"
            else:
                return f"In the context of IVF treatment: {query}"
        
        # If not clearly IVF-related, don't enhance (will be rejected by guardrail)
        return query
