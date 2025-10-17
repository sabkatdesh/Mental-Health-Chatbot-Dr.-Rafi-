# =============================
# 📘 PDF to FAISS Embedding Pipeline (Local + GPU)
# =============================

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import os

# Load environment (optional, for other vars)
load_dotenv(find_dotenv())


# =============================
# 1️⃣ Load All PDFs
# =============================
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from '{data_path}'")
    return documents

documents = load_pdf_files(DATA_PATH)


# =============================
# 2️⃣ Split Documents into Chunks
# =============================
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    print(f"<UNK> Splitted {len(text_chunks)} documents from '{DATA_PATH}'")
    return text_chunks

text_chunks = create_chunks(documents)


# =========================
# 3️⃣ Local Embedding Model
# =========================
def get_embedding_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name = "intfloat/e5-small-v2",
        model_kwargs = {'device': "cuda"},
        encode_kwargs = {"batch_size": 16}
    )
    print("✅ Loaded local embedding model: intfloat/e5-small-v2 (GPU)")
    return embedding_model

embedding_model = get_embedding_models()


# =============================
# 4️⃣ Create & Save FAISS Vector Store
# =============================
DB_FAISS_PATH = "vectorstore/db_faiss"

if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

print("⚙️  Creating embeddings... (this may take a few minutes)")

db = FAISS.from_documents(text_chunks, embedding_model)

db.save_local(DB_FAISS_PATH)

print(f"✅ Embeddings completed and saved to '{DB_FAISS_PATH}'")
