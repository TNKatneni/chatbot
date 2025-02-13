import os
import openai
import spacy
import nltk
import traceback
import re
import requests
from summarizer import Summarizer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from keybert import KeyBERT
from nltk.tokenize import word_tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load NLP Models
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords

# Initialize KeyBERT
kw_model = KeyBERT()

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

def validate_reference_links(links):
    """Checks if reference links are valid before sending them to the user."""
    validated_links = []
    for link in links:
        try:
            response = requests.head(link, timeout=5)
            if response.status_code == 200:
                validated_links.append(link)
            else:
                print(f"⚠️ Broken Link Detected: {link} (Status: {response.status_code})")
        except requests.RequestException:
            print(f"⚠️ Broken Link Detected: {link} (Failed to Connect)")
    return validated_links

def preprocess_query(user_message):
    """
    Extracts and structures key components of the user query.
    - Preserves named entities (locations, numbers)
    - Identifies key priorities (safety, schools, job market, etc.)
    - Uses KeyBERT to extract meaningful phrases
    - Constructs a cleaned version that retains meaning
    """

    # 1️⃣ Expand contractions
    contractions = {"I'm": "I am", "don't": "do not", "isn't": "is not", "can't": "cannot"}
    for contraction, expanded in contractions.items():
        user_message = user_message.replace(contraction, expanded)

    # 2️⃣ Process text with spaCy
    doc = nlp(user_message)
    
    # Identify named entities (cities, numbers, etc.)
    named_entities = {ent.text for ent in doc.ents}

    # 3️⃣ Extract meaningful keywords using KeyBERT
    keybert_keywords = kw_model.extract_keywords(
        user_message, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=8
    )
    extracted_phrases = [keyword[0] for keyword in keybert_keywords]

    # 4️⃣ Define categories for structured extraction
    budget_terms = {"budget", "price", "cost", "affordable", "expensive", "mortgage"}
    priority_terms = {"safety", "crime", "schools", "education", "job market", "employment", "property taxes"}
    lifestyle_terms = {"parks", "shopping", "hospitals", "transportation", "public transit", "community"}

    # 5️⃣ Initialize structured output
    extracted_info = {
        "budget": None,
        "location": [],
        "priorities": [],
        "lifestyle": [],
    }

    # 6️⃣ Extract structured elements
    for token in doc:
        word = token.text.lower()
        if word in budget_terms:
            extracted_info["budget"] = token.text
        elif word in priority_terms:
            extracted_info["priorities"].append(token.text)
        elif word in lifestyle_terms:
            extracted_info["lifestyle"].append(token.text)
        elif token.text in named_entities:  # Keep recognized place names, numbers, etc.
            extracted_info["location"].append(token.text)

    # Add extracted phrases from KeyBERT
    extracted_info["priorities"].extend(extracted_phrases)

    # 7️⃣ Remove duplicates while preserving order
    def remove_duplicates_preserve_order(seq):
        seen = set()
        new_list = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                new_list.append(x)
        return new_list

    extracted_info["location"] = remove_duplicates_preserve_order(extracted_info["location"])
    extracted_info["priorities"] = remove_duplicates_preserve_order(extracted_info["priorities"])
    extracted_info["lifestyle"] = remove_duplicates_preserve_order(extracted_info["lifestyle"])

    # 8️⃣ Construct final processed query
    structured_query = "Looking for properties"
    if extracted_info["location"]:
        structured_query += f" in {', '.join(extracted_info['location'])}"
    if extracted_info["budget"]:
        structured_query += f" within a {extracted_info['budget']} budget"
    if extracted_info["priorities"]:
        structured_query += f" with priorities in {', '.join(extracted_info['priorities'])}"
    if extracted_info["lifestyle"]:
        structured_query += f", considering lifestyle factors like {', '.join(extracted_info['lifestyle'])}"

    return structured_query.strip()

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

        # Process query with structured extraction
        processed_query = preprocess_query(user_message)
        print(f"Processed Query: {processed_query}")  # Debugging

        # **Updated OpenAI Prompt for Reliable References**
        prompt = f"""
        You are a helpful assistant that provides structured answers in Markdown.
        Answer the user's question concisely with **bold** text and bullet points.

        At the end, always include:
        1. **A section labeled 'References:'**
        2. **Three relevant reference links in a numbered list format (1., 2., 3.)**
        3. **Ensure the sources are real and authoritative from the following list:**
           - **Official government (.gov) sites for crime, tax, or legal information**
           - **Education (.edu) sites for school rankings**
           - **Real estate platforms (Zillow, Redfin, Realtor, GreatSchools, Niche)**
           - **Only return links that actually exist and are valid**
        
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

        if not response.choices:
            print("OpenAI API Error: No response choices received")
            return jsonify({"response": "Error: No response from OpenAI.", "processed_query": processed_query}), 500

        bot_response = response.choices[0].message.content.strip()
        print(f"OpenAI Response: {bot_response}")  # Debugging

        # Extract References
        reference_match = re.search(r"(?i)(\*\*References:\*\*|\nReferences:)", bot_response)

        if reference_match:
            split_index = reference_match.start()
            main_response = bot_response[:split_index].strip()
            references_text = bot_response[split_index:].strip()
            if not references_text.startswith("**References:**"):
                references_text = "**References:**\n" + references_text
        else:
            main_response = bot_response.strip()
            references_text = "**References:**\n_No references found._"

        return jsonify({
            "processed_query": processed_query,
            "response": main_response,
            "references": references_text
        })

    except Exception as e:
        print("Unexpected Error: ", str(e))
        print(traceback.format_exc())
        return jsonify({"response": f"An error occurred: {str(e)}", "processed_query": "N/A"}), 500

if __name__ == "__main__":
    app.run(debug=True)
