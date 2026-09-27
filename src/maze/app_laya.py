import asyncio
import os
import random
from concurrent.futures import ThreadPoolExecutor

# Checkpoints are cached locally after the first run; skip the Hugging Face
# Hub round-trip that otherwise happens on startup to re-validate the cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from laya import Router

# Each predict() call is real CPU compute; pin torch to a single thread per
# call so parallelism (when it happens) comes from our own thread pool
# rather than torch's internal thread pool fighting it. See
# sentiment_analysis/app_laya.py for the full writeup.
torch.set_num_threads(1)

WIDTH = 12
HEIGHT = 12
MAX_STEPS = 400

DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
executor = ThreadPoolExecutor(max_workers=8)

# Loaded once at process startup and reused for every request.
router = Router(preload=True)


def generate_maze(width, height):
    walls = {(x, y): {"N": True, "S": True, "E": True, "W": True} for x in range(width) for y in range(height)}
    visited = {(0, 0)}
    stack = [(0, 0)]

    while stack:
        x, y = stack[-1]
        neighbors = []
        for d, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                neighbors.append((d, nx, ny))
        if not neighbors:
            stack.pop()
            continue
        d, nx, ny = random.choice(neighbors)
        walls[(x, y)][d] = False
        walls[(nx, ny)][OPPOSITE[d]] = False
        visited.add((nx, ny))
        stack.append((nx, ny))

    return walls


def open_directions(walls, cell):
    return [d for d, blocked in walls[cell].items() if not blocked]


def call_laya(state, questions):
    return router.predict(state, questions)


async def choose_direction(x, y, goal, candidates):
    """candidates: list of (direction, nx, ny, distance_to_goal). Returns (direction, probabilities, confidence)."""
    gx, gy = goal
    lines = [
        f"A snake is solving a maze on a grid. It is at cell ({x},{y}). "
        f"The goal cell is at ({gx},{gy}). Below are the directions it can move "
        f"right now, each with the resulting cell and how many grid steps "
        f"(Manhattan distance) would remain to the goal after that move. "
        f"Choose the direction that makes the most sense as a maze-solving move."
    ]
    for d, nx, ny, dist in candidates:
        lines.append(f"- {d}: moves to ({nx},{ny}), {dist} steps from goal afterwards")
    state = "\n".join(lines)

    questions = {
        "move": {
            "type": "choice",
            "instructions": "Which direction should the snake move next?",
            "criteria": {
                d: f"Move {d} to ({nx},{ny}); {dist} steps from goal afterwards"
                for d, nx, ny, dist in candidates
            },
        }
    }

    loop = asyncio.get_event_loop()
    response_json = await loop.run_in_executor(executor, call_laya, state, questions)
    answer = response_json.get("answers", {}).get("move", {})
    return answer.get("choice"), answer.get("probabilities"), answer.get("confidence")


@app.get("/")
async def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read().replace("{{ENGINE}}", "Laya"))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    walls = generate_maze(WIDTH, HEIGHT)
    start = (0, 0)
    goal = (WIDTH - 1, HEIGHT - 1)
    position = start
    prev = None
    visited = {start}
    steps = 0
    jev_calls = 0

    await websocket.send_json(
        {
            "type": "maze",
            "width": WIDTH,
            "height": HEIGHT,
            "start": {"x": start[0], "y": start[1]},
            "goal": {"x": goal[0], "y": goal[1]},
            "open": {f"{x},{y}": open_directions(walls, (x, y)) for x, y in walls},
        }
    )

    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") != "step" or position == goal:
                continue

            x, y = position
            open_dirs = open_directions(walls, position)
            prev_dir = None
            if prev is not None:
                for d, (dx, dy) in DIRS.items():
                    if (x + dx, y + dy) == prev:
                        prev_dir = d
                        break

            forward = [d for d in open_dirs if d != prev_dir]
            unvisited_forward = [
                d for d in forward if (x + DIRS[d][0], y + DIRS[d][1]) not in visited
            ]
            options = unvisited_forward or forward or ([prev_dir] if prev_dir else [])

            if not options:
                await websocket.send_json({"type": "stuck", "steps": steps})
                continue

            probabilities = None
            confidence = None
            forced = len(options) == 1

            if forced:
                direction = options[0]
            else:
                gx, gy = goal
                candidates = []
                for d in options:
                    dx, dy = DIRS[d]
                    nx, ny = x + dx, y + dy
                    dist = abs(gx - nx) + abs(gy - ny)
                    candidates.append((d, nx, ny, dist))
                direction, probabilities, confidence = await choose_direction(x, y, goal, candidates)
                jev_calls += 1
                if direction not in options:
                    direction = min(candidates, key=lambda c: c[3])[0]

            dx, dy = DIRS[direction]
            new_position = (x + dx, y + dy)
            prev = position
            position = new_position
            visited.add(position)
            steps += 1

            reached_goal = position == goal
            await websocket.send_json(
                {
                    "type": "move",
                    "from": {"x": x, "y": y},
                    "to": {"x": position[0], "y": position[1]},
                    "direction": direction,
                    "forced": forced,
                    "probabilities": probabilities,
                    "confidence": confidence,
                    "reachedGoal": reached_goal,
                    "steps": steps,
                    "engineCalls": jev_calls,
                }
            )

            if reached_goal:
                await websocket.send_json({"type": "done", "steps": steps, "engineCalls": jev_calls})
            elif steps >= MAX_STEPS:
                await websocket.send_json({"type": "stuck", "steps": steps})
    except WebSocketDisconnect:
        pass
