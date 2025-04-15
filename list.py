from TTS.api import TTS

# Create an instance of TTS
tts = TTS()

# List all available models
models = tts.list_models()
for model in models:
    print(model)