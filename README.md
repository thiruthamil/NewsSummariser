# NewsSummariser

# News Summarizer API

A FastAPI-based service that fetches news using NewsAPI and summarizes them using Hugging Face BART.

## Features
- Summarize custom text (`POST /summarize`)
- Get summarized news by category (`GET /news/{category}`)

## Run Locally
```bash
pip install -r requirements.txt
uvicorn newssummarizer_withapi:app --reload
