'''Detects language of input text using langdetect library, with support for multiple languages
and graceful fallback to English if detection fails.'''

from langdetect import detect, LangDetectException
from typing import Optional
from utils.logger import logger


class LanguageDetector:
    """Detect language of text"""

    # Language code to full name mapping
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh-cn': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'ur': 'Urdu',
        'pa': 'Punjabi',
        'nl': 'Dutch',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'pl': 'Polish'
    }

    # Unicode range checks for CJK scripts — langdetect is unreliable for short text
    @staticmethod
    def _script_detect(text: str) -> Optional[str]:
        """Fast script-based detection for CJK and Arabic before langdetect."""
        import unicodedata
        cjk_k, cjk_j, cjk_c, arab, dev = 0, 0, 0, 0, 0
        for ch in text:
            cp = ord(ch)
            if 0xAC00 <= cp <= 0xD7A3 or 0x3130 <= cp <= 0x318F:  # Hangul
                cjk_k += 1
            elif 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:  # Hiragana/Katakana
                cjk_j += 1
            elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:  # CJK unified
                cjk_c += 1
            elif 0x0600 <= cp <= 0x06FF:  # Arabic
                arab += 1
            elif 0x0900 <= cp <= 0x097F:  # Devanagari (Hindi)
                dev += 1
        total = len(text.replace(" ", "")) or 1
        if cjk_k / total > 0.15: return "ko"
        if cjk_j / total > 0.15: return "ja"
        if arab  / total > 0.15: return "ar"
        if dev   / total > 0.10: return "hi"
        # CJK unified — likely Chinese if no Hangul/Kana
        if cjk_c / total > 0.15: return "zh-cn"
        return None

    def detect_language(self, text: str) -> Optional[str]:
        '''Detect lang (Input_text -> lang_code). Script-first for CJK/Arabic/Hindi.'''
        # Priority 1: fast script detection — always runs first, even for short text
        # This prevents the short-query guard in sql_analyst from bypassing CJK detection
        script = self._script_detect(text)
        if script:
            logger.info(f"🌍 Detected language (script): {self.get_language_name(script)} ({script})")
            return script
        # Priority 2: langdetect for Latin-script with pt/ca disambiguation
        try:
            lang_code = detect(text)
            # Fix: langdetect confuses Portuguese with Catalan for short sentences
            # Portuguese markers: ã, õ, ê, ç + common words
            if lang_code == "ca":
                pt_markers = ["ão", "ões", "ção", "ções", "ão", "qual", "são", "não", "é a", "é o"]
                if any(m in text.lower() for m in pt_markers):
                    lang_code = "pt"
                    logger.info("🌍 Detected language: Portuguese (pt/ca disambiguation)")
                    return lang_code
            logger.info(f"🌍 Detected language: {self.get_language_name(lang_code)} ({lang_code})")
            return lang_code
        except LangDetectException:
            logger.warning("⚠️  Could not detect language, defaulting to English")
            return 'en'

    def is_cjk_text(self, text: str) -> bool:
        """Returns True if text contains significant CJK characters — used by sql_analyst
        to bypass the short-query English default for Japanese/Korean/Chinese."""
        return self._script_detect(text) in ("ja", "ko", "zh-cn")

    def get_language_name(self, lang_code: str) -> str:
        '''Lang_code -> Full language name'''
        return self.LANGUAGE_NAMES.get(lang_code, lang_code.upper())

    def is_supported(self, lang_code: str) -> bool:
        '''Check if lang_code is valid or not'''
        return lang_code in self.LANGUAGE_NAMES