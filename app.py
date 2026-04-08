import random
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

RESIDENTIAL_RESPONSES = [
    "Thank you for your inquiry. At this time, we focus on supplying and supporting larger-scale residential projects and work directly with builders and developers. Unfortunately, we're not set up to take on individual small-scale residential jobs.",
    "Thanks for reaching out - we appreciate your interest. Our work is primarily focused on partnering with builders and developers on larger residential projects, so we're not able to take on individual small residential jobs.",
    "Thank you for contacting us. We specialize in supporting builders and developers on larger residential projects, so we're not able to take on individual small-scale residential work. You may want to check with a local contractor or supplier who handles smaller projects."
]

RESIDENTIAL_TRIGGERS = [
    "residential", "home", "house", "my house", "my home", "apartment",
    "condo", "townhouse", "single family", "personal project", "small project",
    "individual", "homeowner", "my room", "my basement", "my bathroom",
    "my kitchen", "my garage", "my bedroom", "diy", "do it yourself"
]

def is_greeting(message: str) -> bool:
    msg = message.lower().strip()
    return any(msg.startswith(trigger) for trigger in GREETING_TRIGGERS)

def is_thank_you(message: str) -> bool:
    msg = message.lower().strip()
    return any(trigger in msg for trigger in THANK_YOU_TRIGGERS)

def is_residential(message: str) -> bool:
    msg = message.lower()
    return any(trigger in msg for trigger in RESIDENTIAL_TRIGGERS)

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

    if is_residential(user_message):
        return jsonify({"answer": random.choice(RESIDENTIAL_RESPONSES)})

    rep_answer = lookup_rep(user_message)
    if rep_answer is not None:
        ack = get_response("acknowledging_a_question")
        return jsonify({"answer": f"{ack}\n\n{rep_answer}"})

    answer = query_rag(user_message)

    if is_fallback(answer):
        return jsonify({"answer": get_response("cannot_answer")})

    ack = get_response("acknowledging_a_question")
    return jsonify({"answer": f"{ack} {answer}"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
