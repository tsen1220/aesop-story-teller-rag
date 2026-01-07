# Fable RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for Aesop's Fables, built with FastAPI and Qdrant vector database.

## Overview

This project implements a semantic search and question-answering system over a collection of Aesop's Fables. It combines vector embeddings for semantic search with multiple LLM providers for natural language generation.

**Key Features:**
- Semantic search across fables using multilingual embeddings
- Multiple LLM provider support (Ollama, Claude, Gemini, Codex)
- High-performance vector similarity search with Qdrant
- RESTful API with automatic OpenAPI documentation
- Comprehensive test coverage (98%)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FastAPI Application                 │
│  ┌────────────────────────────────────────────────┐  │
│  │  Handlers: /search, /generate, /fables, etc   │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  ┌──────────────────┐      ┌──────────────────────┐ │
│  │  Embeddings      │      │   LLM Providers      │ │
│  │  (Sentence-BERT) │      │  Ollama | Claude     │ │
│  └────────┬─────────┘      │  Gemini | Codex      │ │
│           │                └──────────┬───────────┘ │
└───────────┼───────────────────────────┼─────────────┘
            │                           │
            v                           v
    ┌──────────────┐           ┌──────────────┐
    │   Qdrant     │           │  LLM APIs    │
    │  (Vectors)   │           │              │
    └──────────────┘           └──────────────┘
```

## Quick Start

### Prerequisites

- **Python**: 3.12 or higher
- **uv**: Fast Python package installer ([install guide](https://github.com/astral-sh/uv))
- **Docker**: For running Qdrant vector database
- **Ollama** (optional): For local LLM inference

### Installation

1. Clone and setup:
```bash
git clone <repository-url>
cd fable-rag

# Install dependencies with uv
uv sync
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env as needed
```

3. Start Qdrant vector database:
```bash
docker-compose up -d
```

4. Initialize database with fables:
```bash
uv run python -m src.init_database
```

5. Start the API server:
```bash
uv run python -m src.main
# or
uv run uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

View interactive documentation at `http://localhost:8000/docs`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Search Fables

Find fables by semantic similarity:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "story about honesty and truthfulness",
    "limit": 5,
    "score_threshold": 0.5
  }'
```

### Generate Answer (RAG)

Ask questions about fables with context-aware generation:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What can we learn about honesty from fables?",
    "limit": 3,
    "provider": "ollama",
    "ollama_model": "llama3.1:8b"
  }'
```

### Get Fable by ID

```bash
curl http://localhost:8000/fables/{fable_id}
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

All configuration is managed through environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | `fables` | Vector collection name |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model for semantic search |
| `LLM_PROVIDERS` | `ollama,claude_code,gemini_cli,codex` | Comma-separated list of enabled providers |
| `LLM_DEFAULT_PROVIDER` | `ollama` | Default LLM provider |
| `OLLAMA_MODELS` | `llama3.1:8b` | Comma-separated list of Ollama models |
| `RAW_DATA_PATH` | `data/aesop_fables_raw.json` | Raw fables data path |
| `DATA_PATH` | `data/aesop_fables_processed.json` | Processed fables data path |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

## LLM Providers

| Provider | Description | Setup |
|----------|-------------|-------|
| **Ollama** | Local LLM inference | Install [Ollama](https://ollama.ai/) and pull models |
| **Claude Code** | Claude via CLI | Install `claude` CLI tool |
| **Gemini CLI** | Gemini via CLI | Install `gemini` CLI tool |
| **Codex** | Codex via CLI | Install `codex` CLI tool |

## Project Structure

```
fable-rag/
├── src/
│   ├── handlers/              # API route handlers
│   │   ├── fables.py         # Fable retrieval endpoint
│   │   ├── generate.py       # RAG generation endpoint
│   │   ├── health.py         # Health check endpoint
│   │   └── search.py         # Semantic search endpoint
│   ├── llm/                  # LLM provider integrations
│   │   ├── claude_code.py
│   │   ├── codex.py
│   │   ├── gemini_cli.py
│   │   └── ollama.py
│   ├── models/               # Pydantic models
│   │   ├── requests.py       # Request schemas
│   │   └── responses.py      # Response schemas
│   ├── config.py             # Configuration management
│   ├── data_processor.py     # Data processing utilities
│   ├── dependencies.py       # Dependency injection
│   ├── embeddings.py         # Embedding model wrapper
│   ├── init_database.py      # Database initialization
│   ├── main.py              # Application entrypoint
│   └── qdrant_manager.py    # Qdrant client wrapper
├── tests/                    # Unit tests (98% coverage)
├── data/                     # Fables data
│   ├── aesop_fables_raw.json
│   └── aesop_fables_processed.json
├── docker-compose.yml        # Qdrant container config
├── pyproject.toml           # Project configuration
└── README.md
```

## Development

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_main.py -v

# Run with coverage report
uv run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black src tests

# Lint code
uv run flake8 src tests
```

### Test Coverage

Current coverage: **98%**

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

## How It Works

1. **Data Processing**: Fables are loaded from JSON and processed into structured documents
2. **Embedding**: Each fable is converted to a vector embedding using Sentence Transformers
3. **Indexing**: Embeddings are stored in Qdrant vector database with metadata
4. **Search**: User queries are embedded and similar fables are retrieved via cosine similarity
5. **Generation**: Retrieved fables provide context for LLM to generate relevant answers

## License

MIT License
