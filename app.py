import os
import openai
import spacy
import nltk
import traceback
import re
from summarizer import Summarizer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from nltk.tokenize import word_tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load NLP Models
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords

# Initialize Extractive Summarizer
extractive_summarizer = Summarizer()

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Configure API Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"],
)

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Secure API key usage

# Define protected keywords that should not be removed
protected_keywords = {
    "budget", "price", "cost", "rent", "mortgage", "loan", "finance", "payment",
    "crime", "school", "education", "transportation", "public transit",
    "safety", "growth", "market", "appreciation", "investment", "value", "trends"
}

def preprocess_query(user_message):
    """
    Preprocesses the user message:
    - Expands contractions (e.g., "I'm" → "I am")
    - Removes stopwords while keeping protected terms.
    - Uses extractive summarization if the message is too long.
    """
    # Expand contractions
    contractions = {"I'm": "I am", "don't": "do not", "isn't": "is not", "can't": "cannot"}
    for contraction, expanded in contractions.items():
        user_message = user_message.replace(contraction, expanded)
    
    # Tokenize with spaCy
    doc = nlp(user_message)
    
    # Remove stopwords but preserve protected terms
    filtered_tokens = [
        token.text for token in doc
        if token.text.lower() not in stopwords.words("english") or token.text.lower() in protected_keywords
    ]
    cleaned_text = " ".join(filtered_tokens)

    # Apply extractive summarization only if the query is longer than 25 words
    if len(cleaned_text.split()) > 25:
        summary = extractive_summarizer(cleaned_text, min_length=80, max_length=150)
        return summary  # Extractive summarization keeps structure
    
    return cleaned_text  # Return original if short

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")  # Rate limit each IP
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        print(f"User Input: {user_message}")  # Debugging

        if not user_message:
            return jsonify({"response": "Please enter a valid question.", "processed_query": "N/A"}), 400

        # Process query with stopword removal and optional summarization
        processed_query = preprocess_query(user_message)
        print(f"Processed Query: {processed_query}")  # Debugging

        # **First API Call: Get Answer (Including References)**
        prompt = f"""
        You are a helpful assistant that provides structured answers in Markdown.
        Answer the user's question concisely with **bold** text and bullet points.

        At the end, always include:
        1. **A section labeled 'References:'**
        2. **Three relevant reference links in a numbered list format (1., 2., 3.)**
        3. **Ensure the sources are real and authoritative (government sites, major publications, or official data sources).**

        **User Question:** {processed_query}
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Format all responses using Markdown. Keep responses structured, readable, and include references."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )

        # Ensure OpenAI returned a response
        if not response.choices:
            print("OpenAI API Error: No response choices received")
            return jsonify({"response": "Error: No response from OpenAI.", "processed_query": processed_query}), 500

        bot_response = response.choices[0].message.content.strip()
        print(f"OpenAI Response: {bot_response}")  # Debugging

        # **Extract References Properly**
        reference_match = re.search(r"(?i)(References:|Sources:|Relevant Reference Links:)", bot_response)

        if reference_match:
            split_index = reference_match.start()
            main_response = bot_response[:split_index].strip().rstrip("**")
            references_text = bot_response[split_index:].strip()
            print("References Found in First API Call")
        else:
            main_response = bot_response.rstrip("**")  # Remove trailing "**"
            references_text = None  # Will trigger second API call

        # **If No References, Make a Second API Call**
        if references_text is None:
            print("Searching for references 🔎...")
            ref_prompt = f"""
            You are a helpful assistant that provides only references.
            Based on the user's question below, provide exactly three relevant reference links in a numbered list format.
            Only return the references section.

            **User Question:** {processed_query}
            """

            ref_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only references in Markdown format."},
                    {"role": "user", "content": ref_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            if ref_response.choices:
                references_text = ref_response.choices[0].message.content.strip()
            else:
                references_text = "**References:**\n_No references found for this query._"  # Last fallback

        return jsonify({
            "processed_query": processed_query,  # 🟢 Ensure Processed Query is sent in JSON
            "response": main_response,
            "references": references_text
        })

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")  # Debugging
        return jsonify({"response": f"API Error: {str(e)}", "processed_query": "N/A"}), 500
    except Exception as e:
        print("Unexpected Error: ", str(e))
        print(traceback.format_exc())  # Print full error traceback
        return jsonify({"response": f"An error occurred: {str(e)}", "processed_query": "N/A"}), 500

if __name__ == "__main__":
    app.run(debug=True)