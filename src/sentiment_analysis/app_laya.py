import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Checkpoints are cached locally after the first run; skip the Hugging Face
# Hub round-trip that otherwise happens on startup to re-validate the cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from laya import Router

from sentiment_analysis_with_jev import QUESTIONS, load_rows

# Each predict() call is real CPU compute (a transformer forward pass).
# PyTorch has ONE intra-op thread pool, sized to all cores by default — so
# with our outer ThreadPoolExecutor, N worker threads each try to run ops
# across every core at once, oversubscribing the CPU. Capping torch to one
# thread makes the outer worker pool the only source of parallelism instead
# of the two fighting each other. (laya's own `laya serve` CLI exposes the
# same knob as the LAYA_THREADS env var.)
torch.set_num_threads(1)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#9ca3af",
}
MISMATCH_COLOR = "#111827"

# Loaded once at process startup and reused for every request — unlike the
# CLI client, this avoids paying the checkpoint-loading cost on every call.
router = Router(preload=True)

BATCH_SIZE = 16


def extract_sentiment(response_json):
    return response_json.get("answers", {}).get("sentiment", {}).get("choice")


def batch_predict_sentiment(summaries):
    """Classify many summaries in one (or a few) forward passes instead of
    one call per review, via laya's public `Router.predict_batch` (added in
    laya 0.3.6 — an earlier version of this file reached into undocumented
    `laya.common` internals because the API didn't exist yet at the time).
    """
    requests = [{"state": summary, "questions": QUESTIONS} for summary in summaries]
    return router.predict_batch(requests, batch_size=BATCH_SIZE)


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read().replace("{{ENGINE}}", "Laya"))


@app.get("/api/meta")
async def meta():
    rows = load_rows(None)
    return {"total": len(rows)}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            config = await websocket.receive_json()
            limit = int(config.get("limit") or 0) or None
            workers = max(1, int(config.get("workers") or 8))

            rows = load_rows(limit)
            await websocket.send_json({"type": "init", "ids": [row["id"] for row in rows]})

            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=workers)
            semaphore = asyncio.Semaphore(workers)

            async def process_chunk(chunk):
                async with semaphore:
                    summaries = [row["summary"] for row in chunk]
                    batch_results = await loop.run_in_executor(executor, batch_predict_sentiment, summaries)
                    for row, response_json in zip(chunk, batch_results):
                        predicted = (extract_sentiment(response_json) or "").strip().lower()
                        match = predicted == row["golden"]
                        color = COLORS.get(predicted, "#9ca3af") if match else MISMATCH_COLOR
                        await websocket.send_json(
                            {
                                "type": "result",
                                "id": row["id"],
                                "summary": row["summary"],
                                "golden": row["golden"],
                                "predicted": predicted,
                                "match": match,
                                "color": color,
                            }
                        )

            chunks = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
            tasks = [asyncio.create_task(process_chunk(chunk)) for chunk in chunks]
            await asyncio.gather(*tasks)
            executor.shutdown(wait=False)

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
