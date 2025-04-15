from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import torch
import uuid
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# Allow safe serialization for TTS configurations
torch.serialization.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig])

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Set device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TTS model
try:
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print("Error loading TTS model:", e)
    tts_model = None

# Define request model
class TTSRequest(BaseModel):
    text: str
    speaker_wav: str

# Dictionary to store text parts
text_parts = {}
generated_parts = {}

# Function to split text into parts at sentence boundaries
def split_text_by_sentences(text: str, max_length: int = 500) -> list:
    sentences = text.split('.')
    parts = []
    current_part = ""
    
    for sentence in sentences:
        if len(current_part) + len(sentence) + 1 <= max_length:
            current_part += sentence + '.'
        else:
            parts.append(current_part.strip())
            current_part = sentence + '.'
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

# Function to generate an audio part with error handling
def generate_audio_part(text: str, speaker_wav: str, file_path: str):
    try:
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="ar",
            file_path=file_path
        )
    except Exception as e:
        print(f"Error generating audio for text '{text}': {e}")
        raise e

# Endpoint to split text and initialize parts
@app.post("/initialize-voice")
async def initialize_voice(tts_req: TTSRequest):
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")
    
    text = tts_req.text  # Use raw text directly
    print(f"Original Text: {text}")
    
    parts = split_text_by_sentences(text)
    if not parts:
        raise HTTPException(status_code=400, detail="No valid text provided")
    
    request_id = str(uuid.uuid4())
    text_parts[request_id] = parts
    generated_parts[request_id] = []

    # Generate the first part immediately
    first_part_file = f"{request_id}_part1.wav"
    first_part_path = os.path.join("outputs", first_part_file)
    generate_audio_part(parts[0], tts_req.speaker_wav, first_part_path)
    generated_parts[request_id].append(first_part_path)
    
    response = {
        "request_id": request_id,
        "total_parts": len(parts)
    }
    
    return response

# Endpoint to retrieve and generate a specific audio part
@app.get("/get-part/{request_id}/{part_number}")
async def get_part(request_id: str, part_number: int, background_tasks: BackgroundTasks):
    if request_id not in text_parts:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    parts = text_parts[request_id]
    if part_number > len(parts) or part_number < 1:
        raise HTTPException(status_code=404, detail="Part number out of range")
    
    part_file = f"{request_id}_part{part_number}.wav"
    part_path = os.path.join("outputs", part_file)
    
    if part_path not in generated_parts[request_id]:
        generate_audio_part(parts[part_number - 1], "sounds/SpongBob.wav", part_path)
        generated_parts[request_id].append(part_path)
    
    # Start generating the next part immediately
    next_part_number = part_number + 1
    if next_part_number <= len(parts):
        next_part_file = f"{request_id}_part{next_part_number}.wav"
        next_part_path = os.path.join("outputs", next_part_file)
        if next_part_path not in generated_parts[request_id]:
            background_tasks.add_task(
                generate_audio_part, parts[next_part_number - 1], "sounds/SpongBob.wav", next_part_path
            )
    
    return {"part": part_number, "audio_file": f"/outputs/{part_file}"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)