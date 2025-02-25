from fastapi import APIRouter, Response
from controllers import TTSController
from models import TTSRequest, TTSResponse

class TTSRoutes:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
            tags=["TTS"]
        )
        self.router.add_api_route(
            "/health",
            self.health_check,
            methods=["GET"],
            tags=["TTS"]
        )
        self.router.add_api_route(
            "/languages",
            self.get_languages,
            methods=["GET"],
            tags=["TTS"]
        )
        self.router.add_api_route(
            "/generate",
            self.generate_speech,
            methods=["POST"],
            response_model=TTSResponse,
            tags=["TTS"]
        )

    @staticmethod
    async def root():
        return {
            "message": "Multilingual TTS API",
            "version": "1.0.0",
            "endpoints": {
                "/languages": "Get list of supported languages",
                "/generate": "Generate TTS audio",
                "/health": "Check API health"
            }
        }

    @staticmethod
    async def health_check():
        return {"status": "healthy"}

    @staticmethod
    async def get_languages():
        return await TTSController.get_languages()

    @staticmethod
    async def generate_speech(request: TTSRequest):
        return await TTSController.generate_speech(request)

    def get_router(self) -> APIRouter:
        return self.router