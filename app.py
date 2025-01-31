import os
import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")  # Ensure correct folder paths
CORS(app)  # Enable CORS

# Configure API Rate Limiting (e.g., 5 requests per minute per IP)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"],  # Adjust as needed
)

# Load OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

@app.route("/")
def home():
    return render_template("index.html")  # Serve index.html from templates folder

@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")  # Rate limit each IP
def chat():
    try:
        user_message = request.json.get("message", "").strip()

        if not user_message:
            return jsonify({"response": "Please enter a valid question."}), 400

        # OpenAI API call (optimized for conciseness)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Provide brief, direct responses. Avoid unnecessary details."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,  # Lowered to keep responses concise and on-topic
            max_tokens=100  # Keeps it within limit
        )

        # Extract and return chatbot's response
        bot_response = response.choices[0].message.content.strip()
        return jsonify({"response": bot_response})

    except openai.APIError as e:
        return jsonify({"response": f"API Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
