#!/bin/bash

# Local development startup script
# This script starts Qdrant in Docker and runs the backend locally

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Fable RAG System - Local Development"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for Qdrant to be ready
wait_for_qdrant() {
    echo -e "${YELLOW}Waiting for Qdrant to be ready...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            echo -e "${GREEN}Qdrant is ready!${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}Qdrant failed to start within 30 seconds${NC}"
    return 1
}

# Step 1: Check prerequisites
echo ""
echo "Step 1: Checking prerequisites..."

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker${NC}"

if ! command_exists uv; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}  ✓ uv${NC}"

# Step 2: Start Qdrant container
echo ""
echo "Step 2: Starting Qdrant..."

# Check if Qdrant container already exists
if docker ps -a --format '{{.Names}}' | grep -q '^qdrant-fables$'; then
    if docker ps --format '{{.Names}}' | grep -q '^qdrant-fables$'; then
        echo -e "${GREEN}  ✓ Qdrant container already running${NC}"
    else
        echo "  Starting existing Qdrant container..."
        docker start qdrant-fables
    fi
else
    echo "  Creating new Qdrant container..."
    docker run -d \
        --name qdrant-fables \
        -p 6333:6333 \
        -p 6334:6334 \
        -v "$PROJECT_ROOT/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest
fi

wait_for_qdrant

# Step 3: Setup Python environment with uv
echo ""
echo "Step 3: Setting up Python environment..."

cd "$PROJECT_ROOT"

# Sync dependencies (creates venv if needed)
echo "  Syncing dependencies..."
uv sync --quiet
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# Step 4: Check if database is initialized
echo ""
echo "Step 4: Checking database..."

COLLECTION_EXISTS=$(curl -s http://localhost:6333/collections/fables 2>/dev/null | grep -c '"status":"ok"' || true)
if [ "$COLLECTION_EXISTS" -eq 0 ]; then
    echo -e "${YELLOW}  ⚠ Collection 'fables' not found${NC}"
    echo "  Run: uv run python -m src.init_database"
else
    echo -e "${GREEN}  ✓ Collection 'fables' exists${NC}"
fi

# Step 5: Start the backend
echo ""
echo "Step 5: Starting backend server..."
echo ""
printf "${GREEN}==========================================\n"
printf "  Server starting at http://localhost:8000\n"
printf "  API docs at http://localhost:8000/docs\n"
printf "==========================================${NC}\n"
echo ""

# Run uvicorn
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
