import os

import requests
from dotenv import load_dotenv

load_dotenv()

TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]

url = "https://api.typesafe.ai/v1/systemone"
headers = {
    "Authorization": f"Bearer {TYPESAFE_API_KEY}",
    "Content-Type": "application/json",
}
payload = {
    "state": (
        "Hi, I've been trying to connect my Stripe account for 3 days and it "
        "keeps failing. I'm losing sales. Please help ASAP."
    ),
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity",
        },
    },
}

response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()
print(response.json())
