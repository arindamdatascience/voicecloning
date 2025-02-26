# from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
# from pydantic import BaseModel
# import uvicorn
# import os
# import edge_tts
# import asyncio
# from langdetect import detect
# import platform
# from typing import Optional
# import base64
# from fastapi.middleware.cors import CORSMiddleware
# import tempfile
# import os
# import json
# import subprocess
# import shutil
# from fastapi import FastAPI, File, UploadFile
# from glob import glob
# from pathlib import Path

# app = FastAPI()

# TRAINING_STATUS = {"status": "idle"}  # Track training progress

# DATASET_PATH = "dataset/indictts"
# CONFIG_PATH = "configs/xtts_finetune/config.json"
# OUTPUT_PATH = "output/"
# BEST_MODEL_PATH = os.path.join(OUTPUT_PATH, "best_model.pth")


# def setup_environment():
#     """
#     Installs required dependencies for Coqui-TTS.
#     """
#     os.system("pip install coqpit torchaudio torch numpy tqdm phonemizer")
#     os.system("pip install git+https://github.com/coqui-ai/TTS.git")  # Install Coqui-TTS


# @app.post("/upload-dataset/")
# async def upload_dataset(file: UploadFile = File(...)):
#     """
#     API to upload the IndicTTS dataset as a ZIP file.
#     """
#     file_path = os.path.join(DATASET_PATH, file.filename)
    
#     if not os.path.exists(DATASET_PATH):
#         os.makedirs(DATASET_PATH)
    
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     os.system(f"unzip -o {file_path} -d {DATASET_PATH}")
#     os.remove(file_path)

#     return {"message": "Dataset uploaded and extracted successfully"}


# def preprocess_dataset():
#     """
#     Converts dataset audio and transcripts into a format compatible with Coqui-TTS.
#     """
#     train_txt_path = os.path.join(DATASET_PATH, "train.txt")
#     wav_files = glob(os.path.join(DATASET_PATH, "**/*.wav"), recursive=True)
    
#     if not wav_files:
#         return {"error": "No audio files found in dataset!"}

#     with open(train_txt_path, "w", encoding="utf-8") as f:
#         for wav_file in wav_files:
#             txt_file = wav_file.replace(".wav", ".txt")
#             if os.path.exists(txt_file):
#                 with open(txt_file, "r", encoding="utf-8") as t:
#                     transcript = t.read().strip()
#                 f.write(f"{wav_file}|{transcript}|hi\n")

#     return train_txt_path


# def create_config_file(train_txt_path):
#     """
#     Creates a JSON configuration file optimized for training on an 8GB RAM CPU machine.
#     """
#     config_dir = "configs/xtts_finetune"
#     if not os.path.exists(config_dir):
#         os.makedirs(config_dir)

#     config = {
#         "run_name": "xtts_finetune_indic_cpu",
#         "output_path": OUTPUT_PATH,
#         "datasets": [{"path": train_txt_path, "language": "hi"}],
#         "use_cuda": False,  # Run on CPU
#         "num_workers": 0,  # Disable multi-threading
#         "batch_size": 4,  # Reduce memory usage
#         "epochs": 10,  # Limit training duration
#         "learning_rate": 5e-5,
#         "grad_clip": 1.0,
#         "mixed_precision": "fp16",
#         "model": {
#             "name": "xtts_v2",
#             "checkpoint": "tts_models/multilingual/multi-dataset/xtts_v2",
#             "freeze_base": True,
#             "fine_tune_layers": 2
#         }
#     }

#     with open(CONFIG_PATH, "w", encoding="utf-8") as f:
#         json.dump(config, f, indent=4)

#     return CONFIG_PATH


# def run_training():
#     """
#     Runs the XTTS fine-tuning process.
#     """
#     global TRAINING_STATUS
#     TRAINING_STATUS["status"] = "training"

#     train_script = "TTS/bin/train.py"
#     if not os.path.exists(train_script):
#         TRAINING_STATUS["status"] = "error"
#         return {"error": "Training script not found. Clone Coqui-TTS first."}

#     command = ["python", train_script, "--config_path", CONFIG_PATH]

#     try:
#         subprocess.run(command, check=True)
#         TRAINING_STATUS["status"] = "completed"
#         return {"message": "Training completed!"}
#     except subprocess.CalledProcessError as e:
#         TRAINING_STATUS["status"] = "failed"
#         return {"error": str(e)}


# @app.post("/start-training/")
# def start_training():
#     """
#     API endpoint to start training XTTS model.
#     """
#     if TRAINING_STATUS["status"] == "training":
#         return {"error": "Training is already in progress!"}

#     if os.path.exists(BEST_MODEL_PATH):
#         return {"message": "Training already completed, model available."}

#     train_txt_path = preprocess_dataset()
#     if "error" in train_txt_path:
#         return train_txt_path

#     create_config_file(train_txt_path)

#     return run_training()


# @app.get("/status/")
# def get_status():
#     """
#     API endpoint to check training status.
#     """
#     return TRAINING_STATUS


# if __name__ == "__main__":
#     import uvicorn
#     setup_environment()  # Ensure dependencies are installed
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# # # Initialize FastAPI app
# # app = FastAPI(
# #     title="Multilingual TTS API",
# #     description="API for generating text-to-speech in multiple languages",
# #     version="1.0.0"
# # )

# # # Add CORS middleware
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Set up event loop policy for Windows
# # if platform.system() == 'Windows':
# #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# # # Define request models
# # class TTSRequest(BaseModel):
# #     text: str
# #     language_code: Optional[str] = None
# #     voice: Optional[str] = "female"
# #     encoding: Optional[str] = "latin-1"  # Changed default to latin-1 instead of utf-8

# # class TTSResponse(BaseModel):
# #     success: bool
# #     message: str
# #     audio_base64: Optional[str] = None
# #     detected_language: Optional[str] = None

# # # TTS Configuration with both male and female voices
# # LANGUAGE_MODELS = {
# #     # Indian Languages
# #     'hi': {"name": "Hindi", "male_voice": "hi-IN-MadhurNeural", "female_voice": "hi-IN-SwaraNeural"},
# #     'te': {"name": "Telugu", "male_voice": "te-IN-MohanNeural", "female_voice": "te-IN-ShrutiNeural"},
# #     'ta': {"name": "Tamil", "male_voice": "ta-IN-ValluvarNeural", "female_voice": "ta-IN-PallaviNeural"},
# #     'gu': {"name": "Gujarati", "male_voice": "gu-IN-NiranjanNeural", "female_voice": "gu-IN-DhwaniNeural"},
# #     'mr': {"name": "Marathi", "male_voice": "mr-IN-ManoharNeural", "female_voice": "mr-IN-AarohiNeural"},
# #     'bn': {"name": "Bengali", "male_voice": "bn-IN-BashkarNeural", "female_voice": "bn-IN-TanishaaNeural"},
# #     'ml': {"name": "Malayalam", "male_voice": "ml-IN-MidhunNeural", "female_voice": "ml-IN-SobhanaNeural"},
# #     'kn': {"name": "Kannada", "male_voice": "kn-IN-GaganNeural", "female_voice": "kn-IN-SapnaNeural"},
    
# #     # International Languages
# #     'en': {"name": "English", "male_voice": "en-US-ChristopherNeural", "female_voice": "en-US-JennyNeural"},
# #     'es': {"name": "Spanish", "male_voice": "es-ES-AlvaroNeural", "female_voice": "es-ES-ElviraNeural"},
# #     'fr': {"name": "French", "male_voice": "fr-FR-HenriNeural", "female_voice": "fr-FR-DeniseNeural"},
# #     'de': {"name": "German", "male_voice": "de-DE-ConradNeural", "female_voice": "de-DE-KatjaNeural"},
# #     'zh': {"name": "Chinese", "male_voice": "zh-CN-YunxiNeural", "female_voice": "zh-CN-XiaoxiaoNeural"},
# #     'ja': {"name": "Japanese", "male_voice": "ja-JP-KeitaNeural", "female_voice": "ja-JP-NanamiNeural"},
# #     'ko': {"name": "Korean", "male_voice": "ko-KR-InJoonNeural", "female_voice": "ko-KR-SunHiNeural"},
# #     'ru': {"name": "Russian", "male_voice": "ru-RU-DmitryNeural", "female_voice": "ru-RU-SvetlanaNeural"},
# # }

# # # Helper functions
# # def detect_language(text: str) -> str:
# #     """Detect the language of the input text."""
# #     try:
# #         detected = detect(text)
# #         return detected if detected in LANGUAGE_MODELS else 'en'
# #     except:
# #         return 'en'

# # async def generate_voice_with_edge_tts(text: str, output_path: str, voice: str) -> bool:
# #     """Generate voice using Microsoft Edge TTS service."""
# #     try:
# #         communicate = edge_tts.Communicate(text, voice)
# #         await communicate.save(output_path)
# #         return True
# #     except Exception as e:
# #         print(f"Error generating voice with Edge TTS: {e}")
# #         return False

# # def cleanup_temp_file(file_path: str):
# #     """Clean up temporary file in background."""
# #     if os.path.exists(file_path):
# #         try:
# #             os.remove(file_path)
# #             print(f"Cleaned up temporary file: {file_path}")
# #         except Exception as e:
# #             print(f"Error cleaning up temporary file: {e}")

# # # API endpoints
# # @app.get("/")
# # async def root():
# #     """Root endpoint with API information."""
# #     return {
# #         "message": "Multilingual TTS API",
# #         "version": "1.0.0",
# #         "endpoints": {
# #             "/languages": "Get list of supported languages",
# #             "/generate": "Generate TTS audio",
# #             "/health": "Check API health"
# #         }
# #     }

# # @app.get("/health")
# # async def health_check():
# #     """Health check endpoint."""
# #     return {"status": "healthy"}

# # @app.get("/languages")
# # async def get_languages():
# #     """Get list of supported languages."""
# #     return {
# #         "languages": [
# #             {"code": code, "name": info["name"]}
# #             for code, info in LANGUAGE_MODELS.items()
# #         ]
# #     }

# # @app.post("/generate", response_model=TTSResponse)
# # async def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks):
# #     """Generate speech from text."""
# #     try:
# #         # Auto-detect language if not specified
# #         language_code = request.language_code or detect_language(request.text)
# #         if language_code not in LANGUAGE_MODELS:
# #             raise HTTPException(status_code=400, detail=f"Unsupported language code: {language_code}")

# #         # Create temporary file with proper context management
# #         with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
# #             output_path = temp_file.name

# #         # Select voice based on preference
# #         voice_type = request.voice.lower()
# #         voice_key = "female_voice" if voice_type == "female" else "male_voice"
# #         selected_voice = LANGUAGE_MODELS[language_code][voice_key]
        
# #         # Generate speech using Edge TTS (more consistent across languages)
# #         success = await generate_voice_with_edge_tts(request.text, output_path, selected_voice)
        
# #         if not success:
# #             raise HTTPException(status_code=500, detail="Failed to generate speech")

# #         # Read the generated audio file and convert to base64
# #         with open(output_path, "rb") as audio_file:
# #             audio_data = audio_file.read()
# #             # Use latin-1 encoding for base64 representation
# #             try:
# #                 # Apply the encoding (default is now latin-1)
# #                 audio_base64 = base64.b64encode(audio_data).decode(request.encoding)
# #             except UnicodeDecodeError:
# #                 # Fallback to latin-1 if the requested encoding fails
# #                 audio_base64 = base64.b64encode(audio_data).decode('latin-1')

# #         # Schedule cleanup in background
# #         background_tasks.add_task(cleanup_temp_file, output_path)

# #         return TTSResponse(
# #             success=True,
# #             message="Speech generated successfully",
# #             audio_base64=audio_base64,
# #             detected_language=language_code
# #         )

# #     except Exception as e:
# #         # Make sure to clean up in case of errors
# #         if 'output_path' in locals() and os.path.exists(output_path):
# #             background_tasks.add_task(cleanup_temp_file, output_path)
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/generate/stream")
# # async def generate_speech_stream(request: TTSRequest, background_tasks: BackgroundTasks):
# #     """Generate speech and return as audio stream."""
# #     try:
# #         # Auto-detect language if not specified
# #         language_code = request.language_code or detect_language(request.text)
# #         if language_code not in LANGUAGE_MODELS:
# #             raise HTTPException(status_code=400, detail=f"Unsupported language code: {language_code}")

# #         # Create temporary file with proper context management
# #         with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
# #             output_path = temp_file.name

# #         # Select voice based on preference
# #         voice_type = request.voice.lower()
# #         voice_key = "female_voice" if voice_type == "female" else "male_voice"
# #         selected_voice = LANGUAGE_MODELS[language_code][voice_key]
        
# #         # Generate speech using Edge TTS
# #         success = await generate_voice_with_edge_tts(request.text, output_path, selected_voice)
        
# #         if not success:
# #             raise HTTPException(status_code=500, detail="Failed to generate speech")

# #         # Read the generated audio file
# #         with open(output_path, "rb") as audio_file:
# #             audio_data = audio_file.read()

# #         # Schedule cleanup in background
# #         background_tasks.add_task(cleanup_temp_file, output_path)

# #         # Return audio file as streaming response
# #         return Response(
# #             content=audio_data,
# #             media_type="audio/mpeg",
# #             headers={
# #                 "Content-Disposition": f"attachment; filename=speech_{language_code}.mp3"
# #             }
# #         )

# #     except Exception as e:
# #         # Make sure to clean up in case of errors
# #         if 'output_path' in locals() and os.path.exists(output_path):
# #             background_tasks.add_task(cleanup_temp_file, output_path)
# #         raise HTTPException(status_code=500, detail=str(e))

# # if __name__ == "__main__":
# #     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
import os
import json
import shutil
import soundfile as sf
import librosa
import subprocess
from fastapi import FastAPI, File, UploadFile, Form
from glob import glob
from pathlib import Path
import torch
import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI()

DATASET_PATH = "dataset/indictts"
CONFIG_PATH = "configs/xtts_finetune/config.json"
OUTPUT_PATH = "output/"
BEST_MODEL_PATH = os.path.join(OUTPUT_PATH, "best_model.pth")

TRAINING_STATUS = {"status": "idle"}

# Load pre-trained XTTS model
DEVICE = "cpu"
MODEL_PATH = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = None

def setup_environment():
    """
    Installs required dependencies.
    """
    os.system("pip install coqpit torchaudio torch numpy tqdm phonemizer pydub librosa")
    os.system("pip install git+https://github.com/coqui-ai/TTS.git")

def ensure_config_exists():
    """
    Ensures the config.json file exists.
    """
    config_dir = os.path.dirname(CONFIG_PATH)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if not os.path.exists(CONFIG_PATH):
        config = {
            "run_name": "xtts_finetune_indic_cpu",
            "output_path": OUTPUT_PATH,
            "datasets": [{"path": DATASET_PATH, "language": "hi"}],
            "use_cuda": False,
            "num_workers": 0,
            "batch_size": 2,
            "epochs": 5,
            "learning_rate": 5e-5,
            "grad_clip": 1.0,
            "mixed_precision": "fp16",
            "model": {
                "name": "xtts_v2",
                "checkpoint": MODEL_PATH,
                "freeze_base": True,
                "fine_tune_layers": 2
            }
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

def load_xtts_model():
    """
    Load the fine-tuned XTTS model.
    """
    global tts_model
    if tts_model is None:
        config = XttsConfig()
        config.load_json(CONFIG_PATH)
        tts_model = Xtts.init_from_config(config)
        tts_model.load_checkpoint(BEST_MODEL_PATH, eval=True, device=DEVICE)

@app.post("/clone-voice/")
async def clone_voice(audio_sample: UploadFile = File(...), text: str = Form(...)):
    """
    Clone voice using a given sample and text.
    """
    ensure_config_exists()
    load_xtts_model()

    # Save the uploaded voice sample
    sample_path = "sample.wav"
    with open(sample_path, "wb") as buffer:
        shutil.copyfileobj(audio_sample.file, buffer)

    # Load audio for voice cloning
    audio, sr = librosa.load(sample_path, sr=24000)
    sf.write(sample_path, audio, sr)

    # Generate speech
    output_wav_path = "output_speech.wav"
    tts_model.synthesize(text, sample_path, output_wav_path, language="hi")

    return {"message": "Voice cloned successfully", "output_file": output_wav_path}

if __name__ == "__main__":
    import uvicorn
    setup_environment()
    ensure_config_exists()
    uvicorn.run(app, host="0.0.0.0", port=8000)
