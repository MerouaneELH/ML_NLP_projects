from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DATA_DIR = Path("data/raw")
MIN_LEN = 30 
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "study_index"

# 1) Try the directory loader first (fast)
print("Loading documents from data/raw...")
docs = PyPDFDirectoryLoader(str(DATA_DIR)).load()

# 2) If nothing/very small, fall back to PyMuPDF per file (more robust text extraction)
if len(docs) == 0 or sum(len(d.page_content or "") for d in docs) < 200:
    print("Falling back to PyMuPDFLoader for robust text extraction.")
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        docs.extend(PyMuPDFLoader(str(pdf)).load())

if len(docs) == 0:
    raise RuntimeError("No PDF documents found or loaded.")

# 3) Split
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)

# 4) Filter out empty/short chunks
clean = [c for c in chunks if c.page_content and len(c.page_content.strip()) >= MIN_LEN]

if len(clean) == 0:
    raise RuntimeError(
        "All chunks were empty/too short. Your PDFs may be scanned images.\n"
        "Try using OCR (e.g., install Tesseract) or different PDFs."
    )

# 5) Embed & persist
print(f"Embedding and persisting {len(clean)} chunks...")
emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist the vector store
Chroma.from_documents(
    clean,
    emb,
    persist_directory=VECTORSTORE_DIR,
    collection_name=COLLECTION_NAME 
)

print(f"✅ Vector store ready at ./{VECTORSTORE_DIR} (kept {len(clean)} chunks)")