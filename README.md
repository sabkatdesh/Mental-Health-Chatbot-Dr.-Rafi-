# Mental-Health-Chatbot-Dr.-Rafi-
# 🧠 Mental Health Chatbot 2.0

A conversational AI therapist built with **Python**, **LangChain**, **Groq**, and **Streamlit**, designed to simulate empathetic and supportive therapy-style dialogue.  
The chatbot can **listen**, **think**, and **speak back**, creating a fully interactive voice-based experience.

---

## 🚀 Features

- 🎙️ **Speech-to-Text (STT)**: Converts your voice into text using **Groq Whisper** for fast and accurate transcription.  
- 💬 **AI Therapist Engine**: Uses a **LangChain-powered LLM** backend (`psychological_therapist_chatbot`) to provide meaningful, empathetic responses.  
- 🔊 **Text-to-Speech (TTS)**: Two modes:
  - **Groq PlayAI** – high-quality, expressive voices.
  - **gTTS (Google Text-to-Speech)** – 100% free fallback option.
- 💻 **Frontend (Streamlit)**: Clean, ChatGPT-like chat interface where you can see past conversation history.
- 🪶 Lightweight, simple to deploy on **AWS EC2**, **local machine**, or **Docker**.

---

## 📸 Screenshots
<img width="1890" height="907" alt="brave_screenshot" src="https://github.com/user-attachments/assets/26efa6bf-11d4-4e61-b0f7-53429727b27c" />

---

## 🧩 Project Structure
Mental Health Chatbot 2.0/
│
├── app.py # Streamlit frontend (main UI)
├── app_launch.py # Optional launcher script
│
├── phycological_therapist_brain.py # Core LLM logic (LangChain memory, model, prompt)
├── speech_to_text.py # Handles voice input using Groq Whisper
├── text_to_speech.py # TTS using Groq PlayAI (premium)
├── text_to_speech_gtts.py # Free TTS using gTTS (Google)
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
└── .env # Environment variables (API keys, etc.)


---

## ⚙️ Setup Instructions

### 1. 🧭 Clone the Repository

```bash
git clone https://github.com/sabkatdesh/mental-health-chatbot.git
cd "Mental Health Chatbot 2.0"
```

2. 🧱 Create and Activate a Virtual Environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

3. 📦 Install Dependencies
pip install -r requirements.txt

🔑 Environment Variables
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here
ELEVEN_API_KEY=your_elevenlabs_api_key_here   # optional








