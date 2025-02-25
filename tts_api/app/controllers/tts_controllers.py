import base64
from fastapi import HTTPException
from models import TTSRequest, TTSResponse
from utils import TTSUtils
from config import LANGUAGE_MODELS

class TTSController:
    @staticmethod
    async def generate_speech(request: TTSRequest) -> TTSResponse:
        try:
            language_code = request.language_code or TTSUtils.detect_language(request.text)
            if language_code not in LANGUAGE_MODELS:
                raise HTTPException(status_code=400, detail=f"Unsupported language code: {language_code}")

            # Generate audio as bytes instead of file
            if request.voice.lower() == "male":
                audio_bytes = await TTSUtils.generate_male_voice(request.text, language_code)
            else:
                audio_bytes = TTSUtils.generate_female_voice(request.text, language_code)

            if not audio_bytes:
                raise HTTPException(status_code=500, detail="Failed to generate speech")

            # Convert audio bytes to Base64 string
            audio_base64 = base64.b64encode(audio_bytes).decode()

            return TTSResponse(
                success=True,
                message="Speech generated successfully",
                audio_base64=audio_base64,
                detected_language=language_code
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_languages():
        return {
            "languages": [
                {"code": code, "name": info["name"]}
                for code, info in LANGUAGE_MODELS.items()
            ]
        }
