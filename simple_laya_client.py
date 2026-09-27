import os

# Checkpoints are cached locally after the first run; skip the Hugging Face
# Hub round-trip that otherwise happens on every call to re-validate the
# cache. Unset this (or delete the cache) if you need to fetch new weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from laya import Router

router = Router(preload=True)

state = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it "
    "keeps failing. I'm losing sales. Please help ASAP."
)
questions = {
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
}

result = router.predict(state, questions)
print(result)
