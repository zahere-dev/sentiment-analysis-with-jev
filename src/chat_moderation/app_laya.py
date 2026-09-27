import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Checkpoints are cached locally after the first run; skip the Hugging Face
# Hub round-trip that otherwise happens on startup to re-validate the cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from laya import Router
from openai import AsyncOpenAI

load_dotenv()

# Each predict() call is real CPU compute; pin torch to a single thread per
# call so parallelism (when it happens) comes from our own thread pool
# rather than torch's internal thread pool fighting it. See
# sentiment_analysis/app_laya.py for the full writeup.
torch.set_num_threads(1)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

BLOCK_THRESHOLD = 0.6
MAX_HISTORY_MESSAGES = 20
REFUSAL_MESSAGE = "I'm sorry, but I can't assist with that."

QUESTIONS = {
    # Kept deliberately short — Laya's noul head didn't separate toxic from
    # clean text at all on JEV's long, multi-category instruction (8 bullet
    # points plus explicit non-violation examples); scores came back
    # essentially uncorrelated with actual toxicity. This one-sentence form
    # is closer to Laya's own README examples and gives a clean separation:
    # clean ~0.0-0.16, borderline/civil complaints ~0.42-0.54, actually
    # toxic ~0.74-0.84 (measured on a small manual sample).
    #
    # Tradeoff: narrowing the instruction to just toxicity dropped explicit
    # coverage of prompt injection, PII, self-harm, and spam that the JEV
    # version's longer instruction listed — "ignore previous instructions"
    # scored only 0.10 here since it isn't threatening/hateful language.
    # This version only catches toxic/threatening/harassing/hateful content.
    #"This message is toxic, threatening, harassing, or hateful."
    "toxic": {
        "type": "noul",
        "instructions": (
            "This message is toxic, threatening, harassing, or hateful."
        ),
    }
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
executor = ThreadPoolExecutor(max_workers=8)

# Loaded once at process startup and reused for every request.
router = Router(preload=True)


def call_laya(text):
    return router.predict(text, QUESTIONS)


def extract_toxicity(response_json):
    # Laya's noul answers carry the probability under "noul" (confirmed from
    # its source), unlike JEV's guessed-at "value" key in the JEV version of
    # this app — the two APIs don't share a response shape for this type.
    return response_json.get("answers", {}).get("toxic", {}).get("noul")


async def moderate(text):
    loop = asyncio.get_event_loop()
    response_json = await loop.run_in_executor(executor, call_laya, text)
    return extract_toxicity(response_json)


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read().replace("{{ENGINE}}", "Laya"))


async def send_blocked(websocket, message_id, stage, message=None):
    await websocket.send_json(
        {"type": "blocked", "id": message_id, "stage": stage, "message": message}
    )


async def handle_message(websocket, history, message_id, text):
    moderation_task = asyncio.create_task(moderate(text))
    accumulated = ""
    buffer = []
    cleared = False

    messages = history + [{"role": "user", "content": text}]
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content

        if not cleared and moderation_task.done():
            value = moderation_task.result()
            if value is not None and value >= BLOCK_THRESHOLD:
                await send_blocked(websocket, message_id, "input", REFUSAL_MESSAGE)
                return
            cleared = True
            for buffered in buffer:
                accumulated += buffered
                await websocket.send_json({"type": "token", "id": message_id, "token": buffered})
            buffer.clear()

        if not delta:
            continue

        if cleared:
            accumulated += delta
            await websocket.send_json({"type": "token", "id": message_id, "token": delta})
        else:
            buffer.append(delta)

    if not cleared:
        # The reply finished streaming before moderation resolved (short reply).
        value = await moderation_task
        if value is not None and value >= BLOCK_THRESHOLD:
            await send_blocked(websocket, message_id, "input", REFUSAL_MESSAGE)
            return
        for buffered in buffer:
            accumulated += buffered
            await websocket.send_json({"type": "token", "id": message_id, "token": buffered})
        buffer.clear()

    output_value = await moderate(accumulated) if accumulated else 0
    if output_value is not None and output_value >= BLOCK_THRESHOLD:
        await send_blocked(websocket, message_id, "output")
        return

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": accumulated})
    del history[:-MAX_HISTORY_MESSAGES]

    await websocket.send_json({"type": "done", "id": message_id})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    history = []
    try:
        while True:
            data = await websocket.receive_json()
            await handle_message(websocket, history, data["id"], data["text"])
    except WebSocketDisconnect:
        pass
