from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import query_rag
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"answer": "Please ask a question."}), 400

    answer = query_rag(user_message)
    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
