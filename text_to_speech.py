"""
tts_utils.py
-------------
Groq Text-to-Speech Utility (playai-tts model) with pygame playback
"""

import os
import time
import tempfile
import pygame
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")


def text_to_speech(
    text: str,
    voice: str = "Mason-PlayAI",  # Indian-English style male voice
    format: str = "wav",
    play_audio: bool = True,
):
    """
    Convert text to speech using Groq TTS and play locally.
    """
    if not text.strip():
        print("⚠️ Empty text provided — skipping TTS.")
        return None

    print(f"🎤 Generating voice with Groq ({voice})...")

    client = Groq(api_key=GROQ_API_KEY)

    # create temp path safely
    fd, temp_path = tempfile.mkstemp(suffix=f".{format}")
    os.close(fd)  # close file descriptor immediately

    with client.audio.speech.with_streaming_response.create(
        model="playai-tts",
        voice=voice,
        input=text,
        response_format=format,
    ) as response:
        response.stream_to_file(temp_path)

    # ensure file is flushed and accessible
    time.sleep(0.3)
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"TTS output not found: {temp_path}")

    print(f"✅ Audio generated: {temp_path}")

    if play_audio:
        print("🔊 Playing audio...")
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

    return temp_path


# -------------------------
# 🧪 Test Block
# -------------------------
if __name__ == "__main__":
    sample_text = "Hello, this is a test, using Groq text-to-speech with an Indian English style male voice."
    text_to_speech(sample_text, voice="Mason-PlayAI", format="wav")
