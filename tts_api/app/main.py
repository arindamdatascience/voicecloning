import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import TTSRoutes

app = FastAPI(
    title="Multilingual TTS API",
    description="API for generating text-to-speech in multiple languages",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize and include routes
tts_routes = TTSRoutes()
app.include_router(tts_routes.get_router(), prefix="/api/v1")

# Entry point for running the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Points to the FastAPI instance in this file
        host="0.0.0.0",  # Allows external access; use "127.0.0.1" for local only
        port=8000,  # Port number (change if needed)
        reload=True,  # Auto-reloads server on code changes (useful for development)
        # workers=2  # Number of worker processes (adjust as needed)
    )
