from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer
import requests

# API Key for NewsAPI (replace with your key)
API_KEY = "9c048578b34647e6aa91fb609d76649d"

# Load model once (efficient for serving)
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# FastAPI app
app = FastAPI(title="News Summarizer API")

# Request body format for custom summarization
class NewsRequest(BaseModel):
    text: str
    url: str = None

# Function to fetch news
def fetch_news_by_category(category, country='us', page_size=5):
    url = f"https://newsapi.org/v2/top-headlines?category={category}&country={country}&pageSize={page_size}&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])
    return []

# Function to generate summaries
def generate_summary_with_url(text, url=None, max_length=250, min_length=100):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    sentences = summary.split('. ')[:10]
    summary_with_url = '. '.join(sentences)
    if url:
        summary_with_url += f". [Read more here]({url})"
    return summary_with_url

# API endpoint: fetch & summarize news by category
@app.get("/news/{category}")
def get_news(category: str):
    articles = fetch_news_by_category(category)
    results = []
    for article in articles:
        if article.get("content"):
            summary = generate_summary_with_url(article["content"], article["url"])
            results.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "publishedAt": article["publishedAt"],
                "summary": summary
            })
    return {"category": category, "summaries": results}

# API endpoint: summarize custom text
@app.post("/summarize")
def summarize_text(request: NewsRequest):
    summary = generate_summary_with_url(request.text, request.url)
    return {"summary": summary}
