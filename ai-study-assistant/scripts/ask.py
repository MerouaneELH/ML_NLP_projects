from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
import os 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# --- Configuration ---
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "study_index"
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
# 1) Initialize Embeddings and Vector Store
print("Loading vector store...")
# Using HuggingFaceEmbeddings (modern, non-deprecated)
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=emb,
    collection_name=COLLECTION_NAME # Ensure this matches the builder script
)

# 2) Initialize LLM and QA Chain
print("Initializing LLM and QA chain...")
# Ensure Ollama is running and has the llama3:8b model pulled
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="meta-llama/llama-3-8b-instruct", # You can use any OpenRouter model here
    temperature=0
)
#llm = OllamaLLM(model="llama3:8b")
# from_chain_type uses the retriever to get relevant context
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vs.as_retriever(search_kwargs={"k":4}))

# 3) Invoke Query
query = "whats AI or artificial intelligence and whats the role of a data scientist briefly."
print(f"❓ Query: {query}")
result = qa.invoke({"query": query})

print("\n--- Answer ---")
print(result["result"])
print("--------------")