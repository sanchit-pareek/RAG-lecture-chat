# Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Ollama running locally (or GPU-enabled container)
- Models pulled: `bge-m3`, `llama3.1:8b`

## Quick Start (Development)

### 1. Start Ollama (local)
```bash
ollama serve
ollama pull bge-m3
ollama pull llama3.1:8b
```

### 2. Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`

## Production Deployment (Docker)

### 1. Pull models to Ollama
```bash
ollama pull bge-m3
ollama pull llama3.1:8b
```

### 2. Build and run
```bash
docker-compose up --build
```

The app will be available at `http://localhost`

## Production Deployment (Manual)

### Backend
```bash
cd backend
pip install -r requirements.txt
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend
```bash
cd frontend
npm install
npm run build
# Serve dist/ with nginx or any static file server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama API endpoint |

## GPU Support

For GPU acceleration, ensure:
1. NVIDIA Docker runtime installed
2. `docker-compose.yml` has GPU reservation enabled
