"""
Language detection and translation utilities
Supports Hindi (hi) and English (en), including Hinglish (Roman-script Hindi).
"""
import re
from typing import Optional
from langdetect import detect, LangDetectException
from app.config import settings


class LanguageDetector:
    """Detect and manage language for user messages"""
    
    # Hindi character range (Devanagari script)
    HINDI_PATTERN = re.compile(r'[\u0900-\u097F]')
    
    # Common Hindi words in Devanagari
    HINDI_KEYWORDS = [
        'क्या', 'है', 'में', 'के', 'से', 'को', 'पर', 'नहीं', 'हो', 'गया',
        'कर', 'हैं', 'था', 'थी', 'थे', 'रहे', 'रही', 'रहा', 'होगा', 'होगी'
    ]
    
    # Hinglish: common Hindi words in Roman script (e.g. "ivf kya hai?", "batao")
    # One strong phrase or two+ such words suggests user wants Hindi reply.
    HINGLISH_PHRASES = [
        'kya hai', 'kya hota', 'kya hoti', 'kya hain', 'kyu hai', 'kaise hai',
        'batao', 'bataiye', 'bataye', 'samjhao', 'samjhaiye', 'samjhayen',
        'kya hai?', 'kya hota hai', 'matlab kya', 'kya matlab', 'kya kehte',
    ]
    HINGLISH_WORDS = [
        'kya', 'hai', 'kyu', 'kaise', 'kab', 'kahan', 'kis', 'kaun', 'kuch',
        'koi', 'hota', 'hoti', 'hain', 'batao', 'bataiye', 'samjhao', 'samjhaiye',
        'nahi', 'nahin', 'thik', 'sahi', 'accha', 'achha', 'bilkul', 'ji',
        'hogi', 'hoga', 'karo', 'karein', 'chahiye', 'chahiye?', 'karna',
        'samajh', 'samjh', 'bata', 'bati', 'sab', 'bahut', 'zyada', 'kya?',
    ]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.
        Supports Devanagari Hindi, Hinglish (Roman-script Hindi like "ivf kya hai?"),
        and English. Returns: 'en' or 'hi'
        """
        if not text or not text.strip():
            return settings.DEFAULT_LANGUAGE
        
        # Check for Hindi characters (Devanagari)
        if self.HINDI_PATTERN.search(text):
            return 'hi'
        
        # Check for Hindi keywords (Devanagari)
        text_lower = text.lower().strip()
        hindi_word_count = sum(1 for keyword in self.HINDI_KEYWORDS if keyword in text_lower)
        if hindi_word_count >= 2:
            return 'hi'
        
        # Check for Hinglish phrases (Roman-script Hindi)
        for phrase in self.HINGLISH_PHRASES:
            if phrase in text_lower:
                return 'hi'
        
        # Check for Hinglish words: one strong pair (e.g. "kya" + "hai") or two+ distinct words
        words = set(re.findall(r'\b[a-z]+\b', text_lower))
        hinglish_found = [w for w in self.HINGLISH_WORDS if w in words]
        if len(hinglish_found) >= 2:
            return 'hi'
        if len(hinglish_found) >= 1 and any(q in text_lower for q in ('kya', 'kyu', 'kaise', 'kab', 'kahan', 'batao', 'bataiye', 'samjhao')):
            return 'hi'
        
        # Use langdetect as fallback
        try:
            detected = detect(text)
            if detected == 'hi':
                return 'hi'
            else:
                return 'en'
        except LangDetectException:
            return settings.DEFAULT_LANGUAGE
    
    def is_hindi(self, text: str) -> bool:
        """Check if text contains Hindi"""
        return self.detect_language(text) == 'hi'
    
    def is_english(self, text: str) -> bool:
        """Check if text is English"""
        return self.detect_language(text) == 'en'
    
    def validate_language(self, lang_code: str) -> bool:
        """Validate if language code is supported"""
        return lang_code in settings.SUPPORTED_LANGUAGES


# Global instance
language_detector = LanguageDetector()
