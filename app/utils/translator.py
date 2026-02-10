"""
Translation service for multilingual support
Supports translation between Hindi and English
Uses googletrans (Google Translate unofficial API).
"""
from typing import Optional, Dict
from googletrans import Translator
import logging

logger = logging.getLogger(__name__)


class TranslationService:
    """Handle translation between supported languages"""

    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi'
        }
        self._cache: Dict[str, str] = {}  # Simple in-memory cache

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """
        Translate text to target language

        Args:
            text: Text to translate
            target_lang: Target language code ('en' or 'hi')
            source_lang: Source language code (auto-detect if None)

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        # If target is same as source, return original
        if source_lang and source_lang == target_lang:
            return text

        # Validate target language
        if target_lang not in self.supported_languages:
            logger.warning(f"Unsupported target language: {target_lang}")
            return text

        # Check cache
        cache_key = f"{source_lang or 'auto'}:{target_lang}:{text[:50]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if source_lang:
                result = self.translator.translate(
                    text,
                    src=source_lang,
                    dest=target_lang
                )
            else:
                result = self.translator.translate(text, dest=target_lang)

            translated_text = result.text

            # Cache result
            if len(cache_key) < 200:  # Only cache short keys
                self._cache[cache_key] = translated_text

            return translated_text

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            # Return original text on error
            return text

    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """Translate text to English"""
        return self.translate(text, 'en', source_lang)

    def translate_to_hindi(self, text: str, source_lang: Optional[str] = None) -> str:
        """Translate text to Hindi"""
        return self.translate(text, 'hi', source_lang)

    def batch_translate(self, texts: list, target_lang: str, source_lang: Optional[str] = None) -> list:
        """Translate multiple texts"""
        return [self.translate(text, target_lang, source_lang) for text in texts]

    def clear_cache(self):
        """Clear translation cache"""
        self._cache.clear()


# Global instance
translation_service = TranslationService()
