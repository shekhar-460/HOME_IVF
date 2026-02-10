"""
Intent Classifier - Classifies user intents with multilingual support
"""
from typing import Dict, Optional, List
import re
from app.utils.language_detector import language_detector
from app.utils.translator import translation_service
import logging

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify user intents from messages"""
    
    # Intent patterns for English
    INTENT_PATTERNS_EN = {
        'faq_general': [
            r'what is', r'what are', r'can you tell me', r'explain', r'information about'
        ],
        'faq_process': [
            r'ivf process', r'how does ivf work', r'ivf steps', r'ivf procedure',
            r'how long does', r'what happens', r'ivf cycle'
        ],
        'faq_medication': [
            r'medication', r'medicine', r'drug', r'injection', r'how to take',
            r'when to take', r'dosage', r'prescription'
        ],
        'faq_side_effects': [
            r'side effect', r'adverse', r'problem', r'issue', r'pain', r'discomfort',
            r'nausea', r'bleeding', r'swelling'
        ],
        'appointment_schedule': [
            r'schedule', r'book', r'appointment', r'consultation', r'visit',
            r'available', r'slot', r'timing'
        ],
        'appointment_reschedule': [
            r'reschedule', r'change', r'cancel', r'postpone', r'move'
        ],
        'escalation_urgent': [
            r'emergency', r'urgent', r'severe', r'immediate', r'critical',
            r'chest pain', r'difficulty breathing', r'heavy bleeding'
        ],
        'escalation_complex': [
            r'complicated', r'confused', r'not sure', r'need help', r'understand',
            r'clarify', r'detailed'
        ],
        'greeting': [
            r'hello', r'hi', r'hey', r'good morning', r'good afternoon', r'good evening'
        ],
        'goodbye': [
            r'bye', r'goodbye', r'thank you', r'thanks', r'see you'
        ]
    }
    
    # Intent patterns for Hindi (transliterated and Devanagari)
    INTENT_PATTERNS_HI = {
        'faq_general': [
            r'क्या है', r'क्या हैं', r'बताएं', r'समझाएं', r'जानकारी'
        ],
        'faq_process': [
            r'आईवीएफ प्रक्रिया', r'आईवीएफ कैसे काम करता है', r'आईवीएफ के चरण',
            r'कितना समय लगता है', r'क्या होता है', r'आईवीएफ साइकिल'
        ],
        'faq_medication': [
            r'दवा', r'इंजेक्शन', r'कैसे लें', r'कब लें', r'खुराक'
        ],
        'faq_side_effects': [
            r'साइड इफेक्ट', r'समस्या', r'दर्द', r'परेशानी', r'मतली', r'रक्तस्राव'
        ],
        'appointment_schedule': [
            r'अपॉइंटमेंट', r'बुक करें', r'समय', r'उपलब्ध', r'स्लॉट'
        ],
        'appointment_reschedule': [
            r'बदलें', r'रद्द', r'स्थगित'
        ],
        'escalation_urgent': [
            r'आपातकाल', r'तत्काल', r'गंभीर', r'सीने में दर्द', r'सांस लेने में कठिनाई'
        ],
        'escalation_complex': [
            r'जटिल', r'उलझन', r'समझ नहीं आ रहा', r'मदद चाहिए'
        ],
        'greeting': [
            r'नमस्ते', r'हैलो', r'नमस्कार'
        ],
        'goodbye': [
            r'धन्यवाद', r'शुक्रिया', r'अलविदा'
        ]
    }
    
    def __init__(self):
        self.intents = list(self.INTENT_PATTERNS_EN.keys())
        # Pre-compile regex patterns for better performance (O(1) compilation, O(n) matching)
        self._compiled_patterns_en = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS_EN.items()
        }
        self._compiled_patterns_hi = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS_HI.items()
        }
    
    async def classify(
        self,
        message: str,
        language: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Classify user intent
        
        Args:
            message: User message
            language: Language code ('en' or 'hi'), auto-detect if None
            context: Conversation context
        
        Returns:
            Dict with intent, confidence, and probabilities
        """
        # Detect language if not provided
        if not language:
            language = language_detector.detect_language(message)
        
        # Normalize message
        message_lower = message.lower().strip()
        
        # Get compiled patterns based on language (O(1) lookup)
        compiled_patterns = self._compiled_patterns_hi if language == 'hi' else self._compiled_patterns_en
        patterns = self.INTENT_PATTERNS_HI if language == 'hi' else self.INTENT_PATTERNS_EN
        
        # Calculate scores for each intent (optimized: use compiled patterns, early exit)
        intent_scores = {}
        for intent, pattern_list in compiled_patterns.items():
            score = 0
            # Use compiled patterns for faster matching (O(n) where n is message length)
            for compiled_pattern in pattern_list:
                if compiled_pattern.search(message_lower):  # O(n) - faster than findall
                    score += 0.3  # Weight for each match
                    # Early exit optimization: if we have high confidence, stop checking
                    if score >= 1.0:
                        break
            
            # Cap at 1.0
            if score > 0:
                score = min(score, 1.0)
            
            intent_scores[intent] = score
        
        # If no matches, try translating to English and re-checking (optimized with compiled patterns)
        if max(intent_scores.values()) == 0 and language == 'hi':
            try:
                translated = translation_service.translate_to_english(message, 'hi')
                translated_lower = translated.lower()
                
                # Use compiled patterns for faster matching
                for intent, compiled_pattern_list in self._compiled_patterns_en.items():
                    score = 0
                    for compiled_pattern in compiled_pattern_list:
                        if compiled_pattern.search(translated_lower):  # O(n) - faster than findall
                            score += 0.2  # Lower weight for translated
                            if score >= 1.0:
                                break
                    intent_scores[intent] = max(intent_scores.get(intent, 0), score)
            except Exception as e:
                logger.warning(f"Translation error in intent classification: {e}")
        
        # Find best intent
        if max(intent_scores.values()) > 0:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
        else:
            # Default to general FAQ if no match
            best_intent = 'faq_general'
            confidence = 0.5
        
        # Normalize probabilities
        total_score = sum(intent_scores.values()) or 1
        probabilities = {
            intent: score / total_score
            for intent, score in intent_scores.items()
        }
        
        return {
            'intent': best_intent,
            'confidence': min(confidence, 1.0),
            'all_probabilities': probabilities,
            'language': language
        }
    
    def extract_entities(self, message: str, language: str = "en") -> List[Dict]:
        """Extract entities from message (simplified version)"""
        entities = []
        
        # Date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(today|tomorrow|yesterday)',
            r'(next week|this week)'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': 'DATE',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Medication names (common IVF medications)
        medications = ['gonal', 'menopur', 'cetrotide', 'ovidrel', 'lupron']
        for med in medications:
            if med.lower() in message.lower():
                entities.append({
                    'text': med,
                    'label': 'MEDICATION',
                    'start': message.lower().find(med.lower()),
                    'end': message.lower().find(med.lower()) + len(med)
                })
        
        return entities
