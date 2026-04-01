from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import query_rag, FALLBACK
from dotenv import load_dotenv
from polite_responses import get_response
from rep_lookup import lookup_rep

load_dotenv()

app = Flask(__name__)
CORS(app)

GREETING_TRIGGERS = ["hi", "hello", "hey", "howdy", "good morning", "good afternoon", "good evening", "what's up", "whats up"]
THANK_YOU_TRIGGERS = ["thank you", "thanks", "thx", "thank u", "ty", "appreciate it", "appreciate that"]

def is_greeting(message: str) -> bool:
    msg = message.lower().strip()
    return any(msg.startswith(trigger) for trigger in GREETING_TRIGGERS)

def is_thank_you(message: str) -> bool:
    msg = message.lower().strip()
    return any(trigger in msg for trigger in THANK_YOU_TRIGGERS)

def is_fallback(answer: str) -> bool:
    return answer.strip() == FALLBACK.strip()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"answer": "Please ask a question."}), 400

    if is_greeting(user_message):
        return jsonify({"answer": get_response("greetings")})

    if is_thank_you(user_message):
        return jsonify({"answer": get_response("thanking_the_user")})

    rep_answer = lookup_rep(user_message)
    if rep_answer is not None:
        ack = get_response("acknowledging_a_question")
        return