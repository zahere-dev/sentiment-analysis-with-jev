# Sentiment Analysis with JEV

A collection of small, live demos built around [TypeSafe's JEV](https://docs.typesafe.ai) — a "System One" model designed for fast, structured decisions (`choice`, `score`, `noul`) rather than open-ended text generation. Each demo pairs JEV with a simple real-time UI to show the same idea from a different angle: sentiment classification, content moderation, decision-making in a game, and moderating a live webcam feed.

📺 **Watch the walkthrough:** https://www.youtube.com/watch?v=QoOImcnxfus

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (already gitignored) with:

```
TYPESAFE_API_KEY=your_typesafe_key
OPENAI_API_KEY=your_openai_key   # only needed for chat_moderation and video_moderation
```

## Demos

Each demo is a small FastAPI app under `src/<demo>/`. Run one with:

```bash
cd src/<demo>
../../.venv/bin/uvicorn app:app --reload --port <port>
```

| Demo | Folder | Port | What it shows |
|---|---|---|---|
| **Live Sentiment Grid** | `src/sentiment_analysis` | 8000 | Classifies a batch of real product reviews concurrently; each grid cell flips green/red/gray live as JEV's verdict comes back, and turns black on a mismatch against the human-labeled dataset. Configurable review count and worker concurrency. |
| **Chat Moderation** | `src/chat_moderation` | 8001 | OpenAI streams a chat reply while JEV moderates the input and output in parallel. Clean replies stream through in real time; flagged input is buffered and discarded before the user ever sees it, replaced with a canned refusal. |
| **Snake** | `src/snake` | 8002 | Classic Snake where food is real customer-review text, pre-classified by JEV: positive food grows the snake, neutral gives a small bonus, negative ends the run. |
| **JEV Maze Solver** | `src/maze` | 8003 | A snake navigates a randomly generated maze on its own. At every real junction it asks JEV which direction makes the most progress toward the goal; straight corridors and forced backtracks move without calling the API at all. Live probability bars and a decision log show exactly when and why JEV was consulted. |
| **Video Moderation** | `src/video_moderation` | 8004 | Webcam → GPT-4o-mini (vision) transcribes handwritten signs → JEV classifies the text live as Safe, Toxic, Prompt Injection, or PII. Runs continuously with no manual capture step. |

## Other files

- `simple_jev_client.py` — minimal standalone script showing a raw JEV API call.
- `requirements.txt` — shared dependencies for all demos (fastapi, uvicorn, requests, python-dotenv, openai).
