# Fable RAG System 🦊📚

A Retrieval-Augmented Generation (RAG) system for Aesop's Fables, built with FastAPI, Qdrant vector database, and multiple LLM providers.

## Features

- 🔍 **Semantic Search** - Find fables by meaning, not just keywords
- 🤖 **Multi-LLM Support** - Switch between Ollama, Claude CLI, Gemini CLI, and Codex
- 📊 **Vector Embeddings** - Using `paraphrase-multilingual-MiniLM-L12-v2` for multilingual support
- 🚀 **High Performance** - Qdrant vector database for fast similarity search
- ✅ **Well Tested** - 98% code coverage with comprehensive unit tests

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Application                      │
├──────────────────────────────────────────────────────────────┤
│  Handlers: /search, /generate, /fables, /health              │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────┐  │
│  │   Embedding Model   │    │      LLM Providers          │  │
│  │ (Sentence-Transf.)  │    │ Ollama│Claude│Gemini│Codex  │  │
│  └──────────┬──────────┘    └────────────┬────────────────┘  │
│             │                            │                    │
├─────────────┼────────────────────────────┼────────────────────┤
│             v                            v                    │
│      ┌─────────────┐              ┌───────────┐              │
│      │   Qdrant    │              │  LLM API  │              │
│      │  (Vectors)  │              │           │              │
│      └─────────────┘              └───────────┘              │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (for Qdrant)
- Ollama (optional, for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fable-rag
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env to configure your settings
   ```

4. **Start Qdrant**
   ```bash
   docker-compose up -d
   ```

5. **Initialize the database**
   ```bash
   python -m src.init_database
   ```

6. **Run the API server**
   ```bash
   python -m src.main
   # or
   uvicorn src.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## Configuration

Configuration is managed via environment variables. See `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | `fables` | Vector collection name |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence Transformer model |
| `LLM_PROVIDERS` | `ollama` | Comma-separated list of enabled LLM providers |
| `LLM_DEFAULT_PROVIDER` | `ollama` | Default LLM provider |
| `OLLAMA_MODELS` | `llama3.1:8b` | Comma-separated list of Ollama models |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

## API Endpoints

### Health Check

```bash
GET /health
```

Returns system status and collection info.

### Search Fables

```bash
POST /search
Content-Type: application/json

{
  "query": "story about honesty",
  "limit": 5,
  "score_threshold": 0.5
}
```

### Generate Answer (RAG)

```bash
POST /generate
Content-Type: application/json

{
  "query": "What can we learn about honesty from fables?",
  "limit": 3,
  "provider": "ollama",
  "ollama_model": "llama3.1:8b"
}
```

### Get Fable by ID

```bash
GET /fables/{fable_id}
```

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## LLM Providers

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `ollama` | Local LLM via Ollama | [Ollama](https://ollama.ai/) installed |
| `claude_code` | Claude via CLI | `claude` CLI tool |
| `gemini_cli` | Gemini via CLI | `gemini` CLI tool |
| `codex` | Codex via CLI | `codex` CLI tool |

## Project Structure

```
fable-rag/
├── src/
│   ├── handlers/          # API route handlers
│   │   ├── fables.py      # GET /fables/{id}
│   │   ├── generate.py    # POST /generate
│   │   ├── health.py      # GET /health
│   │   └── search.py      # POST /search
│   ├── llm/               # LLM provider integrations
│   │   ├── claude_code.py
│   │   ├── codex.py
│   │   ├── gemini_cli.py
│   │   └── ollama.py
│   ├── models/            # Pydantic request/response models
│   ├── config.py          # Configuration management
│   ├── dependencies.py    # Dependency injection
│   ├── embeddings.py      # Embedding model wrapper
│   ├── init_database.py   # Database initialization script
│   ├── main.py            # FastAPI application entrypoint
│   └── qdrant_manager.py  # Qdrant client wrapper
├── tests/                 # Unit tests (98% coverage)
├── data/                  # Fables data (JSON)
├── docker-compose.yml     # Qdrant container config
└── pyproject.toml         # Project configuration
```

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_main.py -v
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
flake8 src tests
```

## Test Coverage

```
Name                       Stmts   Miss  Cover
----------------------------------------------
src/__init__.py                1      0   100%
src/config.py                 14      0   100%
src/data_processor.py         39      4    90%
src/dependencies.py           29      0   100%
src/embeddings.py             18      0   100%
src/handlers/__init__.py      11      0   100%
src/handlers/fables.py        18      0   100%
src/handlers/generate.py      45      0   100%
src/handlers/health.py        24      1    96%
src/handlers/search.py        16      0   100%
src/init_database.py          56      0   100%
src/llm/__init__.py            5      0   100%
src/llm/claude_code.py        31      1    97%
src/llm/codex.py              40      2    95%
src/llm/gemini_cli.py         32      1    97%
src/llm/ollama.py             51      0   100%
src/main.py                   27      1    96%
src/models/__init__.py         3      0   100%
src/models/requests.py        11      0   100%
src/models/responses.py       25      0   100%
src/qdrant_manager.py         59      3    95%
----------------------------------------------
TOTAL                        555     13    98%
```

## License

MIT License
