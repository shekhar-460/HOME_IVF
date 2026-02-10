"""
Language detection and translation utilities
Supports Hindi (hi) and English (en)
"""
import re
from typing import Optional
from langdetect import detect, LangDetectException
from app.config import settings


class LanguageDetector:
    """Detect and manage language for user messages"""
    
    # Hindi character range (Devanagari script)
    HINDI_PATTERN = re.compile(r'[\u0900-\u097F]')
    
    # Common Hindi words/phrases
    HINDI_KEYWORDS = [
        'क्या', 'है', 'में', 'के', 'से', 'को', 'पर', 'नहीं', 'हो', 'गया',
        'कर', 'हैं', 'था', 'थी', 'थे', 'रहे', 'रही', 'रहा', 'होगा', 'होगी'
    ]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        Returns: 'en' or 'hi'
        """
        if not text or not text.strip():
            return settings.DEFAULT_LANGUAGE
        
        # Check for Hindi characters
        if self.HINDI_PATTERN.search(text):
            return 'hi'
        
        # Check for Hindi keywords
        text_lower = text.lower()
        hindi_word_count = sum(1 for keyword in self.HINDI_KEYWORDS if keyword in text_lower)
        if hindi_word_count >= 2:
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
