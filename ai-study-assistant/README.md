# AI Study Assistant

A robust, secure AI-powered study assistant that uses Retrieval-Augmented Generation (RAG) to help you query your study materials. Built with FastAPI, LangChain, ChromaDB, and OpenRouter.

## Features

- **Document Ingestion**: Load and index PDF study materials
- **Intelligent Q&A**: Ask questions about your documents using natural language
- **Vector Search**: Fast retrieval using ChromaDB and HuggingFace embeddings
- **Monitoring**: Prometheus metrics and Weights & Biases logging
- **Security**: API key authentication and rate limiting
- **Containerized**: Docker deployment with non-root user

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- OpenRouter API key
- Study materials in PDF format

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ai-study-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your-openrouter-api-key-here
API_KEY=your-secure-api-key-here
```

### Data Preparation

1. Place your PDF files in the `data/raw/` directory

2. Build the vector index:
```bash
python scripts/build_index.py
```

### Running the Application

#### Local Development
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker
```bash
docker build -t ai-study-assistant .
docker run -p 8000:8000 -v $(pwd)/vectorstore:/app/vectorstore ai-study-assistant
```

## API Usage

### Authentication
All API requests require an API key in the Authorization header:
```
Authorization: Bearer your-api-key-here
```

### Endpoints

#### Health Check
```http
GET /health
```

#### Query Documents
```http
POST /query
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "query": "What is machine learning?"
}
```

Response:
```json
{
  "answer": "Machine learning is...",
  "source_documents": [...]
}
```

#### Metrics
```http
GET /metrics
```

### Rate Limiting
- 10 requests per minute per IP address
- Applies to the `/query` endpoint

## Architecture

- **Frontend**: FastAPI web framework
- **RAG Pipeline**: LangChain with ChromaDB vector store
- **Embeddings**: HuggingFace Sentence Transformers
- **LLM**: OpenRouter API (Meta Llama 3)
- **Monitoring**: Prometheus + Weights & Biases
- **Security**: API key auth + rate limiting

## Security Features

- API key authentication
- Input validation and sanitization
- Rate limiting (10 req/min)
- Non-root Docker container
- Environment variable configuration
- Sensitive data exclusion from version control

## Development

### Project Structure
```
ai-study-assistant/
├── api/                    # FastAPI application
│   └── main.py
├── scripts/               # Utility scripts
│   ├── build_index.py    # Document indexing
│   └── ask.py           # CLI query tool
├── data/
│   ├── raw/             # Input PDFs
│   └── processed/       # Processed data
├── vectorstore/         # ChromaDB persistence
├── wandb/              # Experiment tracking
├── Dockerfile
├── requirements.txt
├── .env                # Environment variables (not committed)
└── .gitignore
```

### Testing
```bash
# Run with pytest
pytest

# With coverage
pytest --cov=api --cov-report=html
```

### Monitoring
- Access metrics at `/metrics`
- View W&B logs in the `wandb/` directory
- Prometheus counters for queries and errors

## Configuration

Environment variables:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `API_KEY`: API key for authentication
- `OPENROUTER_MODEL`: LLM model (default: meta-llama/llama-3-8b-instruct)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Troubleshooting

### Common Issues

1. **"Service Unavailable"**: Check if vectorstore is built and models loaded
2. **Authentication errors**: Verify API key in Authorization header
3. **Rate limited**: Wait before making more requests
4. **PDF loading fails**: Ensure PDFs are not password-protected or corrupted

### Logs
Check application logs for detailed error messages. Enable debug logging by setting `LOG_LEVEL=DEBUG` in environment.

## Roadmap

- [ ] Web UI interface
- [ ] Support for additional document formats
- [ ] Multi-user support
- [ ] Advanced RAG techniques (HyDE, multi-query)
- [ ] Integration with cloud storage