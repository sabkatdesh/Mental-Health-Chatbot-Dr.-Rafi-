"""
text_to_speech.py
-----------------
Simple and free Text-to-Speech using Google gTTS + pygame playback.

Usage:
    from text_to_speech import text_to_speech
    text_to_speech("Hello! How are you today?")
"""

import os
import tempfile
import pygame
from gtts import gTTS


def text_to_speech(text: str, lang: str = "en", play_audio: bool = True):
    """
    Convert text to speech using Google gTTS (free).

    Args:
        text (str): The text to speak.
        lang (str): Language code (default "en").
        play_audio (bool): Whether to play the generated audio.

    Returns:
        str: Path to generated audio file.
    """
    if not text.strip():
        print("⚠️ Empty text provided — skipping TTS.")
        return None

    print(f"🎤 Generating voice with gTTS ({lang})...")

    # Generate speech
    tts = gTTS(text=text, lang=lang, slow=False)

    # Save to temporary mp3 file
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    temp_path = temp_audio.name
    temp_audio.close()

    print(f"✅ Audio generated: {temp_path}")

    # Play audio (optional)
    if play_audio:
        print("🔊 Playing audio...")
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

    return temp_path


# 🧪 Test Block
if __name__ == "__main__":
    sample_text = "Hi! This is your AI speaking with a free Google voice."
    text_to_speech(sample_text)
