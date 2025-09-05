from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from transformers import pipeline

# ✅ Read NewsAPI key from environment variable
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize FastAPI
app = FastAPI(title="News Summarizer API")

# ✅ Use a smaller summarization model to fit Render free tier
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6"
)

# Request body schema
class SummarizeRequest(BaseModel):
    text: str

# Function to fetch news from NewsAPI
def fetch_news_by_category(category: str):
    url = f"https://newsapi.org/v2/top-headlines?category={category}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data.get("articles", [])

# Function to safely summarize text
def summarize_text(text: str, max_length: int = 130, min_length: int = 30):
    # ✅ Limit input size to prevent memory issues
    if len(text) > 1000:
        text = text[:1000]
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return summary[0]["summary_text"]

# Endpoint: Summarize custom text
@app.post("/summarize")
def summarize(request: SummarizeRequest):
    summary = summarize_text(request.text)
    return {"summary": summary}

# Endpoint: Get summarized news by category
@app.get("/news/{category}")
def get_news(category: str):
    articles = fetch_news_by_category(category)
    summaries = []

    for article in articles[:5]:  # ✅ Limit to 5 articles
        if article.get("content"):
            summary = summarize_text(article["content"])
            summaries.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "publishedAt": article["publishedAt"],
                "summary": summary,
                "url": article["url"]
            })

    return {"category": category, "summaries": summaries}