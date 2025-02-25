from pydantic import BaseModel
from typing import Optional

class TTSRequest(BaseModel):
    text: str
    language_code: Optional[str] = None
    voice: Optional[str] = "female"

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_base64: Optional[str] = None
    detected_language: Optional[str] = None
