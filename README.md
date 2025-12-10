# Fable RAG System

A fable story retrieval system using vector database and semantic search technology.

## Overview

This project implements a RAG (Retrieval-Augmented Generation) based fable search system that finds relevant fables based on semantic similarity.

### Key Features

- Semantic story search
- Vectorized text storage
- RESTful API interface
- Multilingual embedding model support

### Tech Stack

- **Backend Framework**: FastAPI
- **Vector Database**: Qdrant
- **Embedding Model**: Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Python Package Manager**: UV
- **Containerization**: Docker Compose
- **Testing**: pytest with 97% code coverage

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) (Python package manager)
- Docker & Docker Compose

### 1. Clone the Repository

```bash
git clone <repository-url>
cd story-teller-rag
```

### 2. Configure Environment Variables

Copy the example environment file and adjust as needed:

```bash
cp backend/.env.example backend/.env
```

### 3. Start Vector Database

Start Qdrant vector database using Docker Compose:

```bash
docker compose up qdrant -d
```

### 4. Initialize Database

Run the database initialization script using UV:

```bash
uv run --directory backend python -m src.init_database
```

This step will:
- Load fable story data
- Generate text embeddings
- Insert data into Qdrant database

### 5. Start API Service

#### Option A: Using Docker Compose (Recommended)

```bash
docker compose up -d
```

The API will be available at `http://localhost:8000`.

#### Option B: Using UV (Local Development)

```bash
uv run --directory backend uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test the API

Visit the following URLs to view API documentation:

- Swagger UI: http://localhost:8000/docs

Or test using curl:

```bash
# Health check
curl http://localhost:8000/health

# Search stories
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "a story about honesty", "limit": 3}'
```

## API Endpoints

### GET /health
Check system health status

### POST /search
Search for similar fables

**Request Example**:
```json
{
  "query": "a story about honesty and lying",
  "limit": 5,
  "score_threshold": 0.7
}
```

**Response Example**:
```json
{
  "query": "a story about honesty and lying",
  "results": [
    {
      "id": 1,
      "title": "The Boy Who Cried Wolf",
      "content": "...",
      "moral": "...",
      "score": 0.85,
      "language": "en",
      "word_count": 150
    }
  ],
  "total_results": 1
}
```

### GET /fables/{fable_id}
Get a specific fable by ID

## Development Guide

### Environment Variables

```env
# Qdrant Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=fables

# Embedding Model
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Data Paths
RAW_DATA_PATH=data/aesop_fables_raw.json
DATA_PATH=data/aesop_fables_processed.json

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Install Dependencies

#### Production Dependencies

Install project dependencies using UV:

```bash
uv pip install --directory backend -r backend/requirements.txt
```

#### Development Dependencies

For development and testing, install development dependencies (includes production dependencies):

```bash
uv pip install --directory backend -r backend/requirements-dev.txt
```

This will install:
- All production dependencies
- Testing frameworks (pytest, pytest-cov, pytest-mock, pytest-asyncio)
- Code quality tools (black, flake8)

### Running Tests

Run all unit tests with coverage report:

```bash
cd backend
pytest
```

Run specific test file:

```bash
pytest tests/test_main.py -v
```

Run tests with detailed coverage report:

```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

The project maintains **97% test coverage** with a minimum threshold of 90%.

#### Test Coverage by Module

- `src/embeddings.py`: 100%
- `src/init_database.py`: 100%
- `src/main.py`: 99%
- `src/qdrant_manager.py`: 95%
- `src/data_processor.py`: 90%

View detailed coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Stop Services

```bash
# Stop all containers
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Project Structure

```
story-teller-rag/
├── backend/
│   ├── src/
│   │   ├── main.py              # FastAPI application
│   │   ├── init_database.py     # Database initialization script
│   │   ├── embeddings.py        # Embedding model wrapper
│   │   ├── qdrant_manager.py    # Qdrant manager
│   │   └── data_processor.py    # Data processing utilities
│   ├── tests/                   # Unit tests (97% coverage)
│   │   ├── conftest.py          # Shared test fixtures
│   │   ├── test_main.py         # API endpoint tests
│   │   ├── test_embeddings.py   # Embedding model tests
│   │   ├── test_qdrant_manager.py
│   │   ├── test_init_database.py
│   │   └── test_data_processor.py
│   ├── data/                    # Data directory
│   ├── requirements.txt         # Production dependencies
│   ├── requirements-dev.txt     # Development dependencies
│   ├── pytest.ini               # Pytest configuration
│   ├── .coveragerc              # Coverage configuration
│   ├── Dockerfile
│   └── .env.example
├── docker-compose.yml           # Docker Compose configuration
└── README.md
```

## Troubleshooting

### Qdrant Connection Failed

Check if the Qdrant container is running:

```bash
docker compose ps
```

If not running, start it:

```bash
docker compose up qdrant -d
```

### Collection Does Not Exist

If the API reports that the collection doesn't exist, run the initialization script:

```bash
uv run --directory backend python -m src.init_database
```

### Permission Error

Ensure the `qdrant_storage` directory has the correct read/write permissions.

## License

MIT
