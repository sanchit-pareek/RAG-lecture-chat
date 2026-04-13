# Deployment Guide

## Prerequisites

- Docker & Docker Compose (for local development)
- Railway account (for production deployment)
- Ollama running locally **OR** Groq API key (for LLM)

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

## Production Deployment (Railway)

### 1. Push to GitHub
Make sure your code is on GitHub - Railway deploys from GitHub repos.

### 2. Create Railway Project
1. Go to [railway.app](https://railway.app) and sign in
2. Click **New Project** → **Empty project**

### 3. Add Services

**Backend Service:**
1. Click **New** → **GitHub Repo**
2. Select your repo and the `backend` branch (or main)
3. Set **Root Directory** to `/backend`
4. Add environment variables:
   - `OLLAMA_BASE_URL` - URL to your Ollama instance (e.g., `http://your-server:11434`)
   - `LLM_PROVIDER` - `ollama` or `groq`
   - `GROQ_API_KEY` - (if using Groq)
5. Add a domain (optional)

**Frontend Service:**
1. Click **New** → **GitHub Repo**
2. Select your repo and the `frontend` branch (or main)
3. Set **Root Directory** to `/frontend`
4. Set the `VITE_API_URL` variable to your backend's URL
5. Add a public domain (required for the frontend)

### 4. Networking
- In the frontend service, set `VITE_API_URL` to reference the backend:
  - Format: `https://[backend-service-name].up.railway.app` or your custom domain

### 5. Deploy
Railway will automatically build and deploy when you push to your connected branches.

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama API endpoint |
| LLM_PROVIDER | ollama | LLM provider: `ollama` or `groq` |
| LLM_MODEL | llama3.1:8b | Model for text generation |
| EMBEDDING_MODEL | bge-m3 | Model for embeddings |
| GROQ_API_KEY | - | API key for Groq (if using groq provider) |

## LLM Provider Options

### Ollama (Local)
- Free, runs on your own hardware
- Requires GPU for good performance
- Set `LLM_PROVIDER=ollama`

### Groq (Cloud)
- Free tier available with `llama-3.1-8b-instant`
- Fast inference
- No GPU required
- Set `LLM_PROVIDER=groq` and `GROQ_API_KEY=your_key`

## GPU Support

For local Docker with GPU acceleration:
1. Install NVIDIA Docker runtime
2. `docker-compose.yml` has GPU reservation enabled
