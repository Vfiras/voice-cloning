import requests

url = "https://720d-196-203-207-178.ngrok-free.app/generate-voice"
headers = {"Content-Type": "application/json"}
payload = {
    "text": "Hello, world!",
    "speaker_wav": "sounds/sound4.wav"
}

response = requests.post(url, json=payload, headers=headers, verify=False)
print(response.json())
