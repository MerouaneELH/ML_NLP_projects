import os
import time
import asyncio
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# --- New Imports for OpenRouter/OpenAI Compatibility ---
# Moved to startup event to avoid import issues 
from dotenv import load_dotenv
# --- Prometheus Monitoring ---
from prometheus_client import Counter, Histogram, make_asgi_app
# --- W&B Logging ---
import wandb
# --- LangChain Imports ---
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM  
from langchain_classic.chains import RetrievalQA

load_dotenv()
# --- Configuration (UPDATED) ---
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "study_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- NEW OpenRouter Configuration ---
# Use the environment variable OPENAI_API_KEY which you loaded with dotenv
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "meta-llama/llama-3-8b-instruct" # The fast, free model

# --- Security Configuration ---
API_KEY = os.getenv("API_KEY", "default-key-change-this")  # Set a secure API key in .env
security = HTTPBearer()

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Study Assistant API",
    description="Query your study notes using a local RAG pipeline.",
    version="1.0.0"
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Prometheus Metrics Definitions ---
QUERY_COUNTER = Counter(
    "api_queries_total", 
    "Total number of queries received by the API"
)
QUERY_ERRORS = Counter(
    "api_query_errors_total",
    "Total number of queries that resulted in an error"
)
# Using your Histogram! This is a great metric.
LATENCY_HISTOGRAM = Histogram( 
    "api_query_latency_seconds",
    "Histogram of API query latencies"
)

# --- Pydantic Models (for Request/Response) ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask the study assistant")

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]


# --- Security Dependency ---
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials


# Global flag for CLI mode
CLI_MODE = False

# --- Load Models & Init W&B on Startup ---
@app.on_event("startup")
async def startup_event():
    global CLI_MODE
    if not CLI_MODE:
        print("INFO:     Loading models and RAG chain...")
    # Check for API Key first
    if not OPENROUTER_API_KEY:
        print("FATAL: API Key not found. Please set OPENAI_API_KEY in your environment.")
        app.state.qa_chain = None
        return

    # Import ChatOpenAI here to avoid module-level import issues
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(f"FATAL: ChatOpenAI import failed: {e}")
        app.state.qa_chain = None
        return

    try:
        app.state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        app.state.vector_store = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=app.state.embeddings,
            collection_name=COLLECTION_NAME
        )
        
        app.state.llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY, # Pass the key retrieved from os.getenv()
            temperature=0,
            openai_api_key=OPENROUTER_API_KEY # Redundant, but ensures compatibility 
        )
        
        app.state.qa_chain = RetrievalQA.from_chain_type(
            llm=app.state.llm,
            retriever=app.state.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        
        # Init W&B on startup
        try:  
            # Disable W&B prompts
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_MODE"] = "offline"  # or "disabled" to completely disable
            wandb.init(project="ai-study-assistant", reinit=True)
            print("INFO:     Weights & Biases initialized.")
        except Exception as e:
            print(f"WARNING:  Failed to initialize Weights & Biases: {e}")

        print("INFO:     Models loaded successfully.")
    
    except Exception as e:
        print(f"ERROR:    Failed to load models: {e}")
        app.state.qa_chain = None


# --- API Endpoints ---

@app.get("/health")
async def health():
    if app.state.qa_chain is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Models not loaded")
    return {"status": "ok", "models_loaded": True}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def query(
    query_req: QueryRequest, 
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    The main endpoint to ask a question to the RAG system.
    Requires a valid API key in the Authorization header.
    """
    QUERY_COUNTER.inc()
    start_time = time.time()  
    
    if request.app.state.qa_chain is None:
        QUERY_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG chain not loaded.")

    try:
        query_text = query_req.query
        qa_chain = request.app.state.qa_chain
        
        # This 'await' is the key. It frees the server to handle other
        # requests while it waits for Ollama.
        result = await qa_chain.ainvoke({"query": query_text})
        
        sources = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in result.get('source_documents', [])
        ]
        
        return QueryResponse(
            answer=result['result'],
            source_documents=sources
        )
        
    except Exception as e:
        QUERY_ERRORS.inc()
        print(f"ERROR:    Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # This block runs whether the request succeeded or failed
        latency = time.time() - start_time
        LATENCY_HISTOGRAM.observe(latency) 
        
        # Log to W&B
        if wandb.run is not None:  
            wandb.log({
                "latency_s": latency, 
                "query_length": len(query_req.query)
            })

# Mount the /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

print("INFO:     FastAPI app initialized. Ready to receive requests.")


# --- CLI Interface ---
async def query_api(query_text: str) -> Dict[str, Any]:
    """Query the API internally"""
    if app.state.qa_chain is None:
        return {"error": "RAG chain not loaded"}

    try:
        result = await app.state.qa_chain.ainvoke({"query": query_text})
        sources = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in result.get('source_documents', [])
        ]
        return {
            "answer": result['result'],
            "source_documents": [doc.dict() for doc in sources]
        }
    except Exception as e:
        return {"error": str(e)}


def print_banner():
    """Print a nice banner for the CLI"""
    print("\n" + "="*60)
    print("🤖 AI STUDY ASSISTANT - Interactive Mode")
    print("="*60)
    print("Ask questions about your study materials!")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("="*60 + "\n")


def print_help():
    """Print help information"""
    print("\nCommands:")
    print("  help    - Show this help")
    print("  quit    - Exit the program")
    print("  exit    - Exit the program")
    print("  clear   - Clear the screen")
    print("  <query> - Ask a question\n")


async def interactive_cli():
    """Run interactive CLI mode"""
    print_banner()

    while True:
        try:
            # Get user input
            query = input("❓ Ask me: ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print_help()
                continue
            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue

            # Process query
            print("🤔 Thinking...")
            result = await query_api(query)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("\n" + "="*50)
                print("📝 ANSWER:")
                print("="*50)
                print(result['answer'])

                if result['source_documents']:
                    print("\n" + "="*50)
                    print("📚 SOURCES:")
                    print("="*50)
                    for i, doc in enumerate(result['source_documents'], 1):
                        print(f"\n[{i}] {doc['page_content'][:200]}...")
                        if len(doc['page_content']) > 200:
                            print("   [...truncated...]")

                print("\n" + "-"*60 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Study Assistant")
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    if args.cli:
        # Set CLI mode
        global CLI_MODE
        CLI_MODE = True

        # Run CLI mode
        print("Starting AI Study Assistant in CLI mode...")
        print("Loading models and RAG chain...")

        # Initialize the app (load models)
        loop = asyncio.get_event_loop()

        # Manually trigger startup event
        loop.run_until_complete(startup_event())

        if app.state.qa_chain is None:
            print("❌ Failed to load models. Check your configuration.")
            return

        print("✅ Models loaded successfully!")
        print()

        # Run interactive CLI
        loop.run_until_complete(interactive_cli())
    else:
        # Run server mode
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()