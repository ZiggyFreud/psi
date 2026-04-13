import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import query_rag, FALLBACK, add_to_knowledge
from dotenv import load_dotenv
from polite_responses import get_response
from rep_lookup import lookup_rep

load_dotenv()

app = Flask(__name__)
CORS(app)

ADMIN_TOKEN = "admin332445"

GREETING_TRIGGERS = ["hi", "hello", "hey", "howdy", "good morning", "good afternoon", "good evening", "what's up", "whats up"]
THANK_YOU_TRIGGERS = ["thank you", "thanks", "thx", "thank u", "ty", "appreciate it", "appreciate that"]

RESIDENTIAL_RESPONSES = [
    "Thank you for your inquiry. At this time, we focus on supplying and supporting larger-scale residential projects and work directly with builders and developers. Unfortunately, we're not set up to take on individual small-scale residential jobs. For more information, please email us at info@panelspec.com or call us at 1-800-947-9422.",
    "Thanks for reaching out - we appreciate your interest. Our work is primarily focused on partnering with builders and developers on larger residential projects, so we're not able to take on individual small residential jobs. For more information, please email us at info@panelspec.com or call us at 1-800-947-9422.",
    "Thank you for contacting us. We specialize in supporting builders and developers on larger residential projects, so we're not able to take on individual small-scale residential work. You may want to check with a local contractor or supplier who handles smaller projects. You can also reach us at info@panelspec.com or 1-800-947-9422."
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

def handle_admin_command(message: str):
    """Parse and execute admin commands. Returns a response string."""
    # Strip the token and any leading whitespace/punctuation
    body = message[len(ADMIN_TOKEN):].strip().lstrip(":").strip()

    if not body:
        return (
            "Admin commands:\n"
            "ADD: <text> — store new knowledge in the bot\n"
            "Example: admin332445 ADD: PSI offers custom panel sizes up to 4x10 feet."
        )

    upper = body.upper()

    if upper.startswith("ADD:"):
        content = body[4:].strip()
        if not content:
            return "Please provide text after ADD: to store."
        success = add_to_knowledge(content)
        if success:
            return f"Done. I've stored the following and will use it going forward:\n\n\"{content}\""
        else:
            return "Something went wrong storing that. Please check the server logs."

    # Unrecognized command — still treat the whole body as an ADD
    success = add_to_knowledge(body)
    if success:
        return f"Stored as new knowledge:\n\n\"{body}\""
    else:
        return "Something went wrong storing that. Please check the server logs."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"answer": "Please ask a question."}), 400

    # Admin training route — checked first, before anything else
    if user_message.strip().startswith(ADMIN_TOKEN):
        response = handle_admin_command(user_message.strip())
        return jsonify({"answer": response})

    if is_greeting(user_message):
        return jsonify({"answer": get_response("greetings")})

    if is_thank_you(user_message):
        return jsonify({"answer": get_response("thanking_the_user")})

    if is_residential(user_message):
        return jsonify({"answer": random.choice(RESIDENTIAL_RESPONSES)})

    rep_answer = lookup_rep(user_message)
    if rep_answer is not None:
        return jsonify({"answer": rep_answer})

    answer = query_rag(user_message)
    if is_fallback(answer):
        return jsonify({"answer": get_response("cannot_answer")})

    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
