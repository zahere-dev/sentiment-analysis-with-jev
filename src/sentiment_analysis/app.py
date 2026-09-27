from concurrent.futures import ThreadPoolExecutor

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from sentiment_analysis_with_jev import call_jev, extract_sentiment, load_rows

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#9ca3af",
}
MISMATCH_COLOR = "#111827"


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read().replace("{{ENGINE}}", "Jev"))


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

            async def process(row):
                async with semaphore:
                    response_json = await loop.run_in_executor(executor, call_jev, row["summary"])
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

            tasks = [asyncio.create_task(process(row)) for row in rows]
            await asyncio.gather(*tasks)
            executor.shutdown(wait=False)

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
