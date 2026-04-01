import json
import random

with open("psi_bot_polite_responses.json", "r") as f:
    _data = json.load(f)

_categories = _data["categories"]

def get_response(category: str, name: str = None) -> str:
    responses = _categories.get(category, {}).get("responses", [])
    if not responses:
        return ""
    text = random.choice(responses)
    if name:
        text = text.replace("{name}", name)
    return text