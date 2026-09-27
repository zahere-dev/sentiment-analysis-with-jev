import asyncio
import csv
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]

API_URL = "https://api.typesafe.ai/v1/systemone"
CSV_PATH = "archive/sentiment.csv"
MAX_TEXT_LEN = 70

QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment of this product review summary?",
        "criteria": {
            "positive": "The reviewer is satisfied or praises the product",
            "negative": "The reviewer is dissatisfied or criticizes the product",
            "neutral": "The reviewer is neither clearly satisfied nor dissatisfied",
        },
    }
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
executor = ThreadPoolExecutor(max_workers=16)


def load_summaries():
    summaries = set()
    with open(CSV_PATH, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary = row["Summary"].strip()
            if summary and len(summary) <= MAX_TEXT_LEN:
                summaries.add(summary)
    return list(summaries)


SUMMARIES = load_summaries()


def call_jev(text):
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"state": text, "model": "jev-latest", "questions": QUESTIONS}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def extract_sentiment(response_json):
    return response_json.get("answers", {}).get("sentiment", {}).get("choice")


async def classify(text):
    loop = asyncio.get_event_loop()
    response_json = await loop.run_in_executor(executor, call_jev, text)
    return extract_sentiment(response_json)


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read().replace("{{ENGINE}}", "Jev"))


@app.get("/api/food")
async def food(count: int = 20):
    count = max(1, min(count, 100))
    texts = random.sample(SUMMARIES, k=min(count, len(SUMMARIES)))
    return {"items": [{"id": str(uuid.uuid4()), "text": text} for text in texts]}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            items = data.get("items", [])

            async def process(item):
                sentiment = await classify(item["text"])
                await websocket.send_json(
                    {
                        "type": "result",
                        "id": item["id"],
                        "sentiment": (sentiment or "neutral").strip().lower(),
                    }
                )

            await asyncio.gather(*(process(item) for item in items))
            await websocket.send_json({"type": "batch_done"})
    except WebSocketDisconnect:
        pass
