"""
Mental Health Chatbot 🧠
Frontend built with Gradio — runs locally.

Features:
- ChatGPT-style interface (keeps chat history)
- Voice-to-text input (via speech_to_text)
- Text-to-speech output (via gTTS)
- Local-friendly (no external hosting setup required)
"""

import gradio as gr
from phycological_therapist_brain import psychological_therapist_chatbot
from speech_to_text import speech_to_text
from text_to_speech import text_to_speech


# -------------------------------
# Chatbot response logic
# -------------------------------
def chat_response(user_input, history):
    """
    Handles one round of chat (text only).
    """
    if not user_input.strip():
        return history + [["", "Please say or type something!"]]

    try:
        # Get AI reply
        answer = psychological_therapist_chatbot(user_input)

        # Generate audio output
        text_to_speech(answer)

        # Append chat
        history = history + [[user_input, answer]]
        return history

    except Exception as e:
        return history + [[user_input, f"⚠️ Error: {e}"]]


# -------------------------------
# Voice input logic
# -------------------------------
def voice_input(history):
    """
    Capture voice input using your speech_to_text function,
    then generate chatbot response.
    """
    user_input = speech_to_text().strip()
    if not user_input:
        return history + [["", "Didn't catch that. Please try speaking again."]]

    try:
        answer = psychological_therapist_chatbot(user_input)
        text_to_speech(answer)
        history = history + [[user_input, answer]]
        return history
    except Exception as e:
        return history + [[user_input, f"⚠️ Error: {e}"]]


# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks(title="Mental Health Chatbot 🧠") as demo:
    gr.Markdown(
        """
        # 🧘 Mental Health Chatbot  
        Speak or type to your AI therapist.  
        *Your conversations stay private — locally processed.*
        """
    )

    chatbot = gr.Chatbot(
        label="Dr. Rafi (AI Therapist)",
        bubble_full_width=False,
        height=500
    )

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Type your thoughts here and press Enter..."
        )
        btn_send = gr.Button("💬 Send", variant="primary")
        btn_voice = gr.Button("🎤 Speak", variant="secondary")

    # Button actions
    btn_send.click(chat_response, inputs=[txt, chatbot], outputs=[chatbot])
    txt.submit(chat_response, inputs=[txt, chatbot], outputs=[chatbot])
    btn_voice.click(voice_input, inputs=[chatbot], outputs=[chatbot])

    gr.Markdown("---")
    gr.Markdown("💡 *Developed by [Your Name]* | Powered by LangChain + Gradio + gTTS")

# -------------------------------
# Launch locally
# -------------------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
