# 🗣️ Coqui TTS Voice Cloning Backend (Arabic XTTS)

This project is a **FastAPI-based backend service** for generating multilingual speech using the [Coqui TTS library](https://github.com/coqui-ai/TTS), specifically the **XTTS v2** model.

It provides an easy-to-use API that allows for:
- Arabic text-to-speech (TTS) generation
- Speaker cloning using reference `.wav` files
- Audio streaming split into parts
- Background task support for long text

> ⚠️ **This is not the official Coqui TTS repository**. This project wraps and extends Coqui's [`xtts_v2`](https://github.com/coqui-ai/TTS) model for specific Arabic-based voice cloning use cases.

---

## 🚀 Features

- 🧠 Arabic text-to-speech (using XTTS v2)
- 🎤 Clone speaker voice from a reference `.wav`
- ✂️ Smart sentence-level text splitting for long input
- ⚙️ Background task audio generation (1st part sync, rest async)
- 📁 Audio parts stored locally (`outputs/`)
- 🔍 Part status tracking via API

---

## 📦 Tech Stack

- Python 3.10+
- FastAPI
- PyTorch
- Coqui TTS (`xtts_v2` model)
- Uvicorn

---

## 📁 API Endpoints

### `POST /initialize-voice`
Initialize generation of multiple audio parts from a text.

#### Example:
```json
{
  "text": "مرحبًا! هذا مثال لاختبار تحويل النص إلى كلام.",
  "speaker_wav": "path_or_url_to_reference_voice.wav"
}
