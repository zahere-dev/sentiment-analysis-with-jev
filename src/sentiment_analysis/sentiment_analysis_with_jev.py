import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]

API_URL = "https://api.typesafe.ai/v1/systemone"
CSV_PATH = "archive/sentiment.csv"

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


def build_payload(summary):
    return {
        "state": summary,
        "model": "jev-latest",
        "questions": QUESTIONS,
    }


def call_jev(summary):
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(API_URL, headers=headers, json=build_payload(summary))
    response.raise_for_status()
    return response.json()


def extract_sentiment(response_json):
    return response_json.get("answers", {}).get("sentiment", {}).get("choice")


def load_rows(limit):
    with open(CSV_PATH, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            summary = row["Summary"].strip()
            golden = row["Sentiment"].strip().lower()
            if not summary or not golden:
                continue
            rows.append({"id": str(len(rows) + 1), "summary": summary, "golden": golden})
            if limit and len(rows) >= limit:
                break
    return rows


def evaluate_row(row):
    response_json = call_jev(row["summary"])
    predicted = (extract_sentiment(response_json) or "").strip().lower()
    return row, predicted


def main():
    parser = argparse.ArgumentParser(description="Validate JEV sentiment predictions against golden labels.")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to evaluate (default: 10, 0 = all rows)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads (default: 8)")
    args = parser.parse_args()

    rows = load_rows(args.limit or None)

    correct = 0
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_row, row) for row in rows]
        for future in as_completed(futures):
            row, predicted = future.result()
            results[row["id"]] = (row, predicted)

    for row in rows:
        _, predicted = results[row["id"]]
        match = predicted == row["golden"]
        correct += match

        print(f"[{row['id']}/{len(rows)}] summary={row['summary']!r}")
        print(f"    golden={row['golden']!r} predicted={predicted!r} match={match}")

    total = len(rows)
    accuracy = correct / total if total else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2%})")


if __name__ == "__main__":
    main()
