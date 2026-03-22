# RAG: Lecture Search System

A Retrieval-Augmented Generation (RAG) system for searching and querying lecture content using embeddings and local LLM.

## Project Overview

This project extracts audio from lecture videos, transcribes them using Whisper, creates embeddings, and provides a query interface to search through lecture content using semantic similarity.

## Features

- 🎥 Extract audio from MP4 videos using FFmpeg
- 🎤 Transcribe audio using faster-whisper with CUDA support
- 🧠 Generate embeddings using bge-m3 model
- 🔍 Semantic search through lecture content
- 📊 Similar segment retrieval with context expansion
- 🤖 LLM integration for intelligent querying (via Ollama)

## Project Structure

```
.
├── videos/                 # Input video files (MP4)
├── audios/                 # Extracted audio files (MP3)
├── transcripts/            # Transcribed lecture data (JSON)
├── process_videos.py       # Extract audio from videos
├── transcription.py        # Transcribe audio to JSON
├── preprocessjson.py       # Create embeddings and index
├── llmquering.py          # Query interface
├── lecture_index.joblib    # Cached embeddings and index
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

### Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

**Additional Requirements:**
- FFmpeg (for audio extraction)
- Ollama with bge-m3 model installed locally

### Installation Steps

1. **Install FFmpeg**
   - Windows: `choco install ffmpeg` or download from ffmpeg.org
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

2. **Set up Ollama**
   ```bash
   ollama pull bge-m3  # For embeddings
   ```
   Ensure Ollama is running on `http://localhost:11434`

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### Step 1: Extract Audio from Videos
```bash
python process_videos.py
```
Converts MP4 videos to MP3 audio files in the `audios/` directory.

### Step 2: Transcribe Audio
```bash
python transcription.py
```
Transcribes audio files using Whisper and saves JSON transcripts with segments in the `transcripts/` directory.

### Step 3: Create Embeddings
```bash
python preprocessjson.py
```
Processes transcripts and creates embeddings using bge-m3 model. Outputs cached index to `lecture_index.joblib`.

### Step 4: Query Lectures
```bash
python llmquering.py
```
Provides an interface to search and query lecture content using semantic similarity.

## Usage

### Query Example
```python
# In llmquering.py
query = "What is machine learning?"
# The system finds relevant lecture segments
```

## Output Formats

### Transcript JSON Structure
```json
{
  "lecture": "lec01",
  "language": "en",
  "duration": 3600.5,
  "segments": [
    {
      "start": 0.0,
      "end": 15.5,
      "text": "Lecture content..."
    }
  ]
}
```

## Performance Notes

- **GPU Required:** faster-whisper with `device="cuda"` requires a CUDA-capable GPU
- **Model:** Uses `large-v3` Whisper model (~3GB)
- **Embeddings:** bge-m3 model runs locally via Ollama
- **Caching:** Embeddings are cached in `lecture_index.joblib` for fast queries

## Troubleshooting

- **CUDA not found:** Install appropriate NVIDIA CUDA toolkit or set `device="cpu"` in transcription.py
- **Ollama connection error:** Ensure Ollama is running: `ollama serve`
- **FFmpeg not found:** Add FFmpeg to PATH or install system-wide

## License

MIT

## Notes

- Ensure sufficient disk space for videos and audio files
- Processing large lectures may take considerable time
- GPU acceleration significantly speeds up transcription
