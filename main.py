import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# Register safe globals for model checkpoint loading.
torch.serialization.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig])

from TTS.api import TTS
import gradio as gr
from pyngrok import ngrok

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set your actual ngrok authtoken here
ngrok.set_auth_token("2sqhmKnE6Yfun65pHWoD6NAszME_5tDwwpSBvbJz9N8Na7RYG")

def generate_voice(text, speaker_wav):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    language = "ar"
    output_path = "outputs/output4.wav"
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_path)
    return output_path

# Create the Gradio interface using the new components
interface = gr.Interface(
    fn=generate_voice,
    inputs=[
        gr.components.Textbox(lines=2, placeholder="Enter text here...", label="Text"),
        gr.components.Textbox(value="sounds/sound4.wav", label="Speaker Audio Path")
    ],
    outputs=gr.components.Audio(type="filepath", label="Generated Audio"),
    title="Text-to-Speech with TTS and ngrok",
    description="Enter text and a speaker audio sample path to clone a voice."
)

port = 7860
public_url = ngrok.connect(port)
print("ngrok tunnel available at:", public_url)

interface.launch(server_name="0.0.0.0", server_port=port, share=False)
