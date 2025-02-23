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
from urllib.parse import urlparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load NLP models
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords

# Initialize KeyBERT
kw_model = KeyBERT()

# Initialize Summarizer
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
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_url(url):
    """
    Removes trailing punctuation such as quotes, commas, periods,
    and parentheses from the URL.
    """
    url = url.strip()
    url = re.sub(r'[\'"(),.]+$', '', url)
    return url

def get_display_text(url):
    """
    Extracts a domain name (like 'zillow.com') from a URL.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url

def check_link(link):
    """
    Tries HEAD, then GET to check if the link is valid.
    """
    try:
        response = requests.head(link, timeout=5)
        if response.status_code in [200, 301, 302]:
            return True
        else:
            response = requests.get(link, timeout=5)
            if response.status_code in [200, 301, 302]:
                return True
            else:
                print(f"⚠️ Link check failed for {link} (Status: {response.status_code})")
    except requests.RequestException:
        print(f"⚠️ Link check failed for {link} (Failed to Connect)")
    return False

def validate_reference_links(links):
    """
    Returns up to 3 valid links from the input list.
    """
    validated_links = []
    for link in links:
        if check_link(link):
            validated_links.append(link)
        if len(validated_links) >= 3:
            break
    return validated_links

def get_valid_references(initial_urls, query):
    """
    Tries to validate the initial GPT-provided URLs. If fewer than 3
    valid links are found, calls GPT again for additional references.
    Returns (validated_urls, source).
    """
    initial_urls = [clean_url(url) for url in initial_urls]
    validated_urls = validate_reference_links(initial_urls)
    source = "initial"

    if len(validated_urls) < 3:
        additional_prompt = f"""
        Please provide three additional reference links in a numbered list format that are directly relevant to the following real estate query:
        "{query}"
        The links must be valid, real, and pertain to crime statistics, school rankings, or real estate information.
        Format each link as a complete URL.
        """
        additional_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Format your answer in Markdown. Include raw HTML anchor tags with target='_blank' for each link."},
                {"role": "user", "content": additional_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        additional_bot_response = additional_response.choices[0].message.content.strip()
        additional_raw_urls = re.findall(r'(https?://\S+)', additional_bot_response)
        additional_urls = [clean_url(url) for url in additional_raw_urls]
        additional_validated = validate_reference_links(additional_urls)
        if additional_validated:
            source = "additional"
        for url in additional_validated:
            if url not in validated_urls:
                validated_urls.append(url)
            if len(validated_urls) >= 3:
                break

    return validated_urls[:3], source

def preprocess_query(user_message):
    """
    Uses spaCy and KeyBERT to extract key info from the user's query.
    Builds a structured query for real estate or location-based questions.
    Also detects if it's a comparison.
    """
    lower_query = user_message.lower()
    comparison_phrases = [
        "better to live",
        "compare",
        "vs",
        "versus",
        "difference between",
        "differences",
        "pros and cons"
    ]
    is_comparison = any(phrase in lower_query for phrase in comparison_phrases)

    # Expand contractions
    contractions = {"I'm": "I am", "don't": "do not", "isn't": "is not", "can't": "cannot"}
    for contraction, expanded in contractions.items():
        user_message = user_message.replace(contraction, expanded)

    doc = nlp(user_message)
    named_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

    keybert_keywords = kw_model.extract_keywords(
        user_message, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=8
    )
    extracted_phrases = [k[0] for k in keybert_keywords]

    extracted_info = {
        "budget": None,
        "location": [],
        "priorities": [],
        "lifestyle": [],
    }
    budget_terms = {"budget", "price", "cost", "affordable", "expensive", "mortgage"}
    priority_terms = {"safety", "crime", "schools", "education", "job market", "employment", "property taxes"}
    lifestyle_terms = {"parks", "shopping", "hospitals", "transportation", "public transit", "community"}

    # Identify budget, location, priorities, lifestyle
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            extracted_info["budget"] = ent.text

    for token in doc:
        lower_word = token.text.lower()
        if lower_word in budget_terms and not extracted_info["budget"]:
            extracted_info["budget"] = token.text
        elif lower_word in priority_terms:
            extracted_info["priorities"].append(token.text)
        elif lower_word in lifestyle_terms:
            extracted_info["lifestyle"].append(token.text)
        elif token.text in named_entities:
            extracted_info["location"].append(token.text)

    # Merge city + state if needed
    merged_locations = []
    skip_next = False
    for i, loc in enumerate(extracted_info["location"]):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(extracted_info["location"]):
            next_item = extracted_info["location"][i+1]
            if next_item.upper() in [
                "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
                "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
                "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
                "TX","UT","VT","VA","WA","WV","WI","WY"
            ]:
                merged_locations.append(loc + " " + next_item)
                skip_next = True
            else:
                merged_locations.append(loc)
        else:
            merged_locations.append(loc)
    extracted_info["location"] = merged_locations

    location_lower = [loc.lower() for loc in extracted_info["location"]]
    final_phrases = [phrase for phrase in extracted_phrases if phrase.lower() not in location_lower]
    extracted_info["priorities"].extend(final_phrases)

    # Remove duplicates while preserving order
    def remove_duplicates_preserve_order(seq):
        seen = set()
        new_list = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                new_list.append(x)
        return new_list

    for k in ["location", "priorities", "lifestyle"]:
        extracted_info[k] = remove_duplicates_preserve_order(extracted_info[k])

    # Build structured query
    if is_comparison and len(extracted_info["location"]) >= 2:
        if "cost" in lower_query or "living" in lower_query:
            structured_query = f"Compare cost of living between {extracted_info['location'][0]} and {extracted_info['location'][1]}."
        else:
            structured_query = f"Compare living conditions between {extracted_info['location'][0]} and {extracted_info['location'][1]}."
    elif "suburb" in lower_query or "suburbs" in lower_query:
        structured_query = "Recommend the best suburbs in Illinois"
        if extracted_info["budget"]:
            structured_query += f" within a {extracted_info['budget']} budget"
        if extracted_info["priorities"]:
            structured_query += f" with priorities in {', '.join(extracted_info['priorities'])}"
        if extracted_info["lifestyle"]:
            structured_query += f", considering lifestyle factors like {', '.join(extracted_info['lifestyle'])}"
    elif any(keyword in lower_query for keyword in ["property", "real estate", "home"]):
        structured_query = "Looking for properties"
        if extracted_info["location"]:
            structured_query += f" in {', '.join(extracted_info['location'])}"
        if extracted_info["budget"]:
            structured_query += f" within a {extracted_info['budget']} budget"
        if extracted_info["priorities"]:
            structured_query += f" with priorities in {', '.join(extracted_info['priorities'])}"
        if extracted_info["lifestyle"]:
            structured_query += f", considering lifestyle factors like {', '.join(extracted_info['lifestyle'])}"
    else:
        if extracted_info["location"] or extracted_info["priorities"]:
            structured_query = "Looking for information regarding"
            if extracted_info["location"]:
                structured_query += " " + ", ".join(extracted_info["location"])
            if extracted_info["priorities"]:
                structured_query += " with focus on " + ", ".join(extracted_info["priorities"])
        else:
            structured_query = user_message

    return structured_query.strip(), is_comparison

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        compare_input_1 = request.json.get("compare_input_1", "").strip()
        compare_input_2 = request.json.get("compare_input_2", "").strip()
        print(f"User Input: {user_message}")

        # Detect "Compare Now" usage: if both compare_input_1 and compare_input_2 are set
        is_compare_flow = bool(compare_input_1 and compare_input_2)

        if not user_message and not is_compare_flow:
            return jsonify({"response": "Please enter a valid question.", "processed_query": "N/A"}), 400

        # Preprocess query
        processed_query, is_comparison = preprocess_query(user_message)
        print(f"Processed Query: {processed_query}")

        # If user specifically says "recommend," add a directive.
        if "recommend" in user_message.lower():
            processed_query += " Please provide direct recommendations."

        # If compare flow or is_comparison, add explicit side-by-side comparison instructions
        if is_compare_flow or is_comparison:
            # If user used the "Compare Now" fields, override processed_query with a direct compare statement
            if is_compare_flow:
                # Build a direct compare statement from the two inputs
                processed_query = (
                    f"Compare {compare_input_1} with {compare_input_2} in a side-by-side format. "
                    "Focus on cost of living, real estate, schools, and any key differences. "
                    "Please provide a direct comparison with bullet points."
                )
            else:
                # If user typed something like "Compare X and Y" in the text box
                processed_query += (
                    " Please provide a direct side-by-side comparison of these two locations "
                    "focusing on cost of living, real estate prices, crime rates, schools, "
                    "job markets, and any key factors."
                )

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
                {"role": "system", "content": "Format all responses using Markdown. Keep responses structured, readable, and include references. Do NOT add extra horizontal lines (---)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )

        if not response.choices:
            print("OpenAI API Error: No response choices received")
            return jsonify({"response": "Error: No response from OpenAI.", "processed_query": processed_query}), 500

        bot_response = response.choices[0].message.content.strip()
        print("OpenAI Response (raw):")
        print(bot_response)

        if is_compare_flow:
            match = re.search(r"(?i)\*\*References:\*\*", bot_response)
            if match:
                bot_response = bot_response[:match.start()].strip()

        bot_response = re.sub(r'^[\s-]*---[\s-]*$', '', bot_response, flags=re.MULTILINE).strip()

        # 1) Separate main response from references
        reference_match = re.search(r"(?i)(\*\*References:\*\*|\nReferences:)", bot_response)
        if reference_match:
            split_index = reference_match.start()
            main_response = bot_response[:split_index].strip()
            references_text = bot_response[split_index:].strip()

            if not references_text.startswith("**References:**"):
                references_text = "**References:**\n" + references_text.lstrip("References:").strip()
        else:
            main_response = bot_response.strip()
            references_text = ""

        # 2) Extract raw URLs
        raw_urls = re.findall(r'(https?://\S+)', references_text)

        # 3) If GPT provided at least 3 URLs, just use them; else fix them
        if len(raw_urls) >= 3:
            selected_urls = raw_urls[:3]
            references_html = """
                <div class="references-container">
                    <strong class="references-title">References:</strong>
                    <div class="reference-buttons">
            """
            for url in selected_urls:
                url_clean = clean_url(url)
                display_text = get_display_text(url_clean)
                references_html += f'<a class="reference-button" href="{url_clean}" target="_blank" rel="noopener noreferrer">{display_text}</a>\n'
            references_html += "</div></div>"
        else:
            validated_urls, ref_source = get_valid_references(raw_urls, processed_query)
            if len(validated_urls) == 3:
                references_html = """
                    <div class="references-container">
                        <strong class="references-title">References:</strong>
                        <div class="reference-buttons">
                """
                for url in validated_urls:
                    url_clean = clean_url(url)
                    display_text = get_display_text(url_clean)
                    references_html += f'<a class="reference-button" href="{url_clean}" target="_blank" rel="noopener noreferrer">{display_text}</a>\n'
                references_html += "</div></div>"
            else:
                references_html = """
                    <div class="references-container">
                        <strong class="references-title">References:</strong>
                        <p>No valid references found.</p>
                    </div>
                """

        final_response = main_response.strip()
        final_references = references_html.strip()

        print("Final Main Response:")
        print(final_response)
        print("\nFinal References:")
        print(final_references)

        return jsonify({
            "processed_query": processed_query,
            "response": final_response,
            "references": final_references
        }), 200, {'Content-Type': 'text/html'}

    except Exception as e:
        print("Unexpected Error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "response": f"An error occurred: {str(e)}",
            "processed_query": "N/A"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
