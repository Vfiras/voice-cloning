from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import torch
import uuid
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # Import XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig  # New import
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Allowlist all required globals for PyTorch 2.6+
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])  # Add XttsArgs
logger.info("Added XttsConfig, XttsAudioConfig, BaseDatasetConfig, and XttsArgs to safe globals for torch.load")

app = FastAPI()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load TTS model
tts_model = None
try:
    logger.info("Attempting to load TTS model...")
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    logger.info("TTS model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TTS model: {e}")
    tts_model = None

class TTSRequest(BaseModel):
    text: str
    speaker_wav: str

text_parts = {}
generated_parts = {}

def split_text_by_sentences(text: str, max_length: int = 500) -> list:
    try:
        sentences = text.split('.')
        parts = []
        current_part = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(current_part) + len(sentence) + 1 <= max_length:
                current_part += sentence + '.'
            else:
                if current_part:
                    parts.append(current_part.strip())
                current_part = sentence + '.' if sentence else ""
        if current_part:
            parts.append(current_part.strip())
        logger.info(f"Text split into {len(parts)} parts")
        return parts
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise

def generate_audio_part(text: str, speaker_wav: str, file_path: str, request_id: str, part_num: int):
    try:
        logger.info(f"Generating part {part_num} for {request_id} at {file_path}")
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="ar",
            file_path=file_path
        )
        if request_id in generated_parts:
            generated_parts[request_id].add(file_path)
        logger.info(f"Finished generating part {part_num} for {request_id}")
    except Exception as e:
        logger.error(f"Error generating part {part_num} for {request_id}: {e}")
        raise

@app.post("/initialize-voice")
async def initialize_voice(tts_req: TTSRequest, background_tasks: BackgroundTasks):
    try:
        if tts_model is None:
            logger.error("TTS model is not loaded")
            raise HTTPException(status_code=500, detail="TTS model not loaded")
        
        text = tts_req.text
        logger.info(f"Original Text: {text}")
        
        parts = split_text_by_sentences(text)
        if not parts:
            logger.error("No valid text provided")
            raise HTTPException(status_code=400, detail="No valid text provided")
        
        request_id = str(uuid.uuid4())
        text_parts[request_id] = parts
        generated_parts[request_id] = set()
        
        os.makedirs("outputs", exist_ok=True)
        logger.info(f"Created outputs directory for {request_id}")
        
        # Generate part 1 synchronously
        first_part_file = f"{request_id}_part1.wav"
        first_part_path = os.path.join("outputs", first_part_file)
        logger.info(f"Starting synchronous generation of part 1 for {request_id}")
        generate_audio_part(parts[0], tts_req.speaker_wav, first_part_path, request_id, 1)
        generated_parts[request_id].add(first_part_path)
        
        # Queue subsequent parts
        for i, part_text in enumerate(parts[1:], 2):
            part_file = f"{request_id}_part{i}.wav"
            part_path = os.path.join("outputs", part_file)
            logger.info(f"Queuing part {i} for {request_id}")
            background_tasks.add_task(generate_audio_part, part_text, tts_req.speaker_wav, part_path, request_id, i)
        
        logger.info(f"Initialized request {request_id} with {len(parts)} parts")
        return {
            "request_id": request_id,
            "total_parts": len(parts)
        }
    except Exception as e:
        logger.error(f"Error in initialize_voice: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/part-status/{request_id}/{part_number}")
async def part_status(request_id: str, part_number: int):
    try:
        if request_id not in text_parts:
            logger.error(f"Request ID {request_id} not found")
            raise HTTPException(status_code=404, detail="Request ID not found")
        if part_number > len(text_parts[request_id]) or part_number < 1:
            logger.error(f"Part number {part_number} out of range for {request_id}")
            raise HTTPException(status_code=404, detail="Part number out of range")
        
        part_file = f"{request_id}_part{part_number}.wav"
        part_path = os.path.join("outputs", part_file)
        if part_path in generated_parts.get(request_id, set()):
            logger.info(f"Part {part_number} for {request_id} is ready")
            return {"status": "done", "audio_url": f"http://172.20.10.4:8001/{part_file}"}
        logger.info(f"Part {part_number} for {request_id} is still pending")
        return {"status": "pending"}
    except Exception as e:
        logger.error(f"Error in part_status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)