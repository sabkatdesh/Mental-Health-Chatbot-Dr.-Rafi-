"""
Enhanced Speech-to-Text using speech_recognition + noise reduction + Groq Whisper
"""

import os
import tempfile
import numpy as np
import speech_recognition as sr
from scipy.io.wavfile import write
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Optional noise reduction
try:
    import noisereduce as nr
except ImportError:
    nr = None  # skip noise reduction if not installed

load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")


def speech_to_text(noise_reduction: bool = True, pause_threshold: float = 3.0) -> str:
    """
    Records voice input until silence, with ambient noise calibration and optional noise reduction.
    Transcribes audio using Groq Whisper.

    Args:
        noise_reduction (bool): Whether to apply background noise reduction (requires noisereduce).
        pause_threshold (float): Seconds of silence before auto-stopping recording.

    Returns:
        str: Transcribed text.
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold  # time of silence before stopping
    print("\n🎙️ Adjusting for ambient noise... Please stay quiet.")
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = max(300, recognizer.energy_threshold * 1.2)  # make it less sensitive to small noise
        print("✅ Calibrated. Start speaking — it will auto-stop after silence.")
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
    print("✅ Recording complete.")

    # Save to temp WAV
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_data = audio.get_wav_data()
    with open(temp_wav.name, "wb") as f:
        f.write(wav_data)

    # Optional noise reduction
    if noise_reduction and nr is not None:
        print("🔇 Applying noise reduction...")
        from scipy.io import wavfile
        rate, data = wavfile.read(temp_wav.name)
        reduced = nr.reduce_noise(y=data, sr=rate)
        write(temp_wav.name, rate, reduced)
    elif noise_reduction:
        print("⚠️ 'noisereduce' not installed, skipping noise filtering.")

    # Transcribe with Groq Whisper
    print("🧠 Transcribing with Groq Whisper...")
    client = Groq(api_key=GROQ_API_KEY)
    with open(temp_wav.name, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
        )

    text = transcription.text.strip()
    if text:
        print(f"🗣️ You said: “{text}”")
    else:
        print("⚠️ No speech detected.")
    return text



