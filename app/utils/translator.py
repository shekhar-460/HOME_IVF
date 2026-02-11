"""
Translation service for multilingual support
Supports translation between Hindi and English (chat) and multiple languages for page translation.
Uses googletrans (Google Translate unofficial API).
"""
from typing import Optional, Dict, List
from googletrans import Translator
import logging

logger = logging.getLogger(__name__)


# Languages offered for page translation (googletrans codes)
PAGE_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "ar": "العربية",
    "bn": "বাংলা",
    "zh-cn": "中文 (简体)",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
}


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
            target_lang: Target language code ('en' or 'hi' for chat; any code for page translate)
            source_lang: Source language code (auto-detect if None)

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        # If target is same as source, return original
        if source_lang and source_lang == target_lang:
            return text

        # Check cache (allow any target for page translation)
        cache_key = f"{source_lang or 'auto'}:{target_lang}:{text[:80]}"
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
            if len(cache_key) < 250:
                self._cache[cache_key] = translated_text

            return translated_text

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """Translate text to English"""
        return self.translate(text, 'en', source_lang)

    def translate_to_hindi(self, text: str, source_lang: Optional[str] = None) -> str:
        """Translate text to Hindi"""
        return self.translate(text, 'hi', source_lang)

    def batch_translate(self, texts: List[str], target_lang: str, source_lang: Optional[str] = None) -> List[str]:
        """Translate multiple texts (for page translation). Empty strings are preserved."""
        if not texts:
            return []
        return [self.translate(text, target_lang, source_lang) if (text and text.strip()) else text for text in texts]

    def clear_cache(self):
        """Clear translation cache"""
        self._cache.clear()


# Global instance
translation_service = TranslationService()
