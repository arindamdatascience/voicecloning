import edge_tts
from gtts import gTTS
from langdetect import detect
from config import LANGUAGE_MODELS
from io import BytesIO

class TTSUtils:
    """Utility class for generating TTS voices and detecting language."""

    @staticmethod
    async def generate_male_voice(text: str, language_code: str) -> bytes:
        """Generate male voice and return audio bytes."""
        try:
            voice = LANGUAGE_MODELS[language_code]["edge_voice"]
            communicate = edge_tts.Communicate(text, voice)
            audio_bytes = await communicate.save_to_bytes()
            return audio_bytes
        except Exception as e:
            print(f"Error generating male voice: {e}")
            return None

    @staticmethod
    def generate_female_voice(text: str, language_code: str) -> bytes:
        """Generate female voice and return audio bytes."""
        try:
            tts = gTTS(text=text, lang=language_code, slow=False)
            return TTSUtils._convert_gtts_to_bytes(tts)
        except Exception as e:
            print(f"Error generating female voice: {e}")
            return None

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the language of the given text."""
        try:
            detected = detect(text)
            return detected if detected in LANGUAGE_MODELS else 'en'
        except:
            return 'en'

    @staticmethod
    def _convert_gtts_to_bytes(tts) -> bytes:
        """Convert gTTS output to in-memory bytes."""
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        return audio_buffer.getvalue()
