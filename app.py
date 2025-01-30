from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return "Chatbot Backend is running."

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    # Return the same message
    bot_response = f'You said: "{user_message}"'
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
