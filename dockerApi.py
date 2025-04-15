from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, torch, uuid, time, threading
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import logging

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Allowlist PyTorch safe globals
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI()

# Serve audio files from "outputs"
app.mount("/audio", StaticFiles(directory="outputs"), name="audio")

device = "cuda" if torch.cuda.is_available() else "cpu"

tts_model = None
try:
    logger.info("Loading TTS model...")
    tts_model = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    config_path=None,
    speaker_wav=None,
    vocoder_path=None,
    vocoder_config_path=None,
    gpu=True,
    agree_to_fine_tune_model_license=True  # <-- this is required!
).to(device)

except Exception as e:
    logger.error(f"Failed to load TTS model: {e}")

class TTSRequest(BaseModel):
    text: str
    speaker_wav: str

text_parts = {}
generated_parts = {}

def split_text_by_sentences(text: str, max_length: int = 500) -> list:
    sentences = text.split('.')
    parts, current_part = [], ""
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
    return parts

def auto_delete(path, delay_seconds=300):
    def delete():
        time.sleep(delay_seconds)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted: {path}")
    threading.Thread(target=delete, daemon=True).start()

def generate_audio_part(text: str, speaker_wav: str, file_path: str, request_id: str, part_num: int):
    try:
        speaker_path = os.path.join("sounds", speaker_wav)
        logger.info(f"Generating part {part_num} for {request_id}")
        tts_model.tts_to_file(text=text, speaker_wav=speaker_path, language="ar", file_path=file_path)
        generated_parts[request_id].add(file_path)
        auto_delete(file_path)
    except Exception as e:
        logger.error(f"Error generating part {part_num}: {e}")
        raise

@app.post("/initialize-voice")
async def initialize_voice(tts_req: TTSRequest, background_tasks: BackgroundTasks):
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model not loaded")

    parts = split_text_by_sentences(tts_req.text)
    if not parts:
        raise HTTPException(status_code=400, detail="Empty text")

    request_id = str(uuid.uuid4())
    text_parts[request_id] = parts
    generated_parts[request_id] = set()

    os.makedirs("outputs", exist_ok=True)
    first_part_file = f"{request_id}_part1.wav"
    first_part_path = os.path.join("outputs", first_part_file)
    generate_audio_part(parts[0], tts_req.speaker_wav, first_part_path, request_id, 1)

    for i, part_text in enumerate(parts[1:], 2):
        part_file = f"{request_id}_part{i}.wav"
        part_path = os.path.join("outputs", part_file)
        background_tasks.add_task(generate_audio_part, part_text, tts_req.speaker_wav, part_path, request_id, i)

    return {"request_id": request_id, "total_parts": len(parts)}

@app.get("/part-status/{request_id}/{part_number}")
async def part_status(request_id: str, part_number: int):
    if request_id not in text_parts:
        raise HTTPException(status_code=404, detail="Invalid request_id")
    if part_number < 1 or part_number > len(text_parts[request_id]):
        raise HTTPException(status_code=404, detail="Invalid part number")

    part_file = f"{request_id}_part{part_number}.wav"
    part_path = os.path.join("outputs", part_file)
    if part_path in generated_parts.get(request_id, set()):
        return {"status": "done", "audio_url": f"/audio/{part_file}"}
    return {"status": "pending"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
