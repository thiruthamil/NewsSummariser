from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from transformers import BartForConditionalGeneration, BartTokenizer

# ✅ Read NewsAPI key from environment variable
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize FastAPI
app = FastAPI(title="News Summarizer API")

# Load summarizer model + tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Request body schema
class SummarizeRequest(BaseModel):
    text: str

# Function to fetch news from NewsAPI
def fetch_news_by_category(category: str):
    url = f"https://newsapi.org/v2/top-headlines?category={category}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data.get("articles", [])

# Function to summarize text
def summarize_text(text: str, max_length: int = 130, min_length: int = 30):
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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

    for article in articles[:5]:  # Limit to 5 articles
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