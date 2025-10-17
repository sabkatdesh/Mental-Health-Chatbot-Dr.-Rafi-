import os
from typing import Any
from dotenv import load_dotenv, find_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain

# -------------------------
# ENV setup
# -------------------------
load_dotenv(find_dotenv())
GROQ_API = os.environ.get("GROQ_API") or os.environ.get("GROQ_API_KEY")

# -------------------------
# Create in-session memory
# -------------------------
def create_session_memory():
    print("🧠 Starting fresh in-memory session.")
    return ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )

# -------------------------
# Load LLM
# -------------------------
def load_llm():
    return ChatGroq(
        temperature=0.4,
        groq_api_key=GROQ_API,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

# -------------------------
# Prompt Template
# -------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are Dr. Rafi, a 40-year-old Bangladeshi clinical psychologist and therapist.

You must:
- Respond calmly, empathetically, and professionally.
- Use verified knowledge strictly from the provided context.
- Integrate relevant therapeutic concepts (CBT, humanistic, trauma-informed) naturally.
- Maintain a comforting Islamic tone when it fits.
- Offer practical, actionable advice when possible.
- Avoid diagnosis unless supported by context.

Here is the conversation so far:
{chat_history}

Context from verified sources:
{context}

User:
{question}

Answer as Dr. Rafi:
"""

def set_custom_prompt(): 
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["chat_history", "context", "question"]
    )

# -------------------------
# Load FAISS DB
# -------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# -------------------------
# Create QA Chain (Modern LCEL syntax)
# -------------------------
def create_qa_chain():
    llm = load_llm()
    prompt = set_custom_prompt()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # LCEL composition
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.get_relevant_documents(x["question"])
        )
        | combine_docs_chain
    )
    return chain

# Initialize chain and memory separately
qa_chain = create_qa_chain()
memory = create_session_memory()

# -------------------------
# Chat Function (manual memory management)
# -------------------------
def psychological_therapist_chatbot(query: str) -> Any:
    # Load chat memory
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    # Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={'k': 3})
    docs = retriever.get_relevant_documents(query)

    # Merge all document texts for the LLM
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Run the LLM chain manually
    response = qa_chain.invoke({
        "question": query,
        "context": context_text,
        "chat_history": chat_history
    })

    # Extract answer
    answer_text = response.get("text", "No response generated.") if isinstance(response, dict) else str(response)

    # Save context to memory
    memory.save_context({"question": query}, {"answer": answer_text})

    # Collect and display source info (filename + page number)
    if docs:
        unique_sources = []
        for doc in docs:
            meta = doc.metadata
            name = meta.get("source", "Unknown Source")
            page = meta.get("page", "N/A")
            unique_sources.append(f"- {os.path.basename(name)} (page {page})")
        sources_text = "\n\n📚 **Sources:**\n" + "\n".join(set(unique_sources))
    else:
        sources_text = "\n\n📚 **Sources:** None found in context."

    return answer_text + sources_text

# -------------------------
# CLI Chat Loop
# -------------------------
if __name__ == "__main__":
    print("🩺 Dr. Rafi is online. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye. Take care of yourself.")
            break

        try:
            answer = psychological_therapist_chatbot(user_input)
            print(f"\nDr. Rafi:\n{answer}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
