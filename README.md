# 🏡 Real Estate Chatbot

Welcome to the **Real Estate Chatbot** project! This chatbot leverages **Natural Language Processing (NLP)** to assist users in finding real estate information based on their queries, including side-by-side location comparisons.

## 📌 Project Overview
This project is part of **MSDS 453: Natural Language Processing**, where we built a chatbot capable of answering real estate-related questions using an NLP pipeline (spaCy, KeyBERT, etc.) and an LLM (OpenAI).

## 👥 Team Members
- **Tarun Katneni**
- **Tina Wang**

## 🚀 Features
- **Conversational AI** for real estate queries  
- **Comparison mode** for side-by-side location comparisons (e.g., cost of living, schools, crime)  
- **Automated reference links** from authoritative real estate or government sites  
- **Built with Flask** for serving the chatbot interface  

## 🛠️ Tech Stack
- **Python** 🐍  
- **Flask** (backend server)  
- **spaCy** (NLP processing)  
- **KeyBERT** (keyword extraction)  
- **OpenAI** API for large language model responses  
- **requests**, **nltk**, **summarizer** (supporting libraries)

---

## ⚙️ Prerequisites
1. **Python 3.8+**  
2. **OpenAI API Key** (set as an environment variable named `OPENAI_API_KEY`)

---

## 💻 Installation & Setup
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/TNKatneni/real-estate-chatbot.git
   cd real-estate-chatbot

2. **Create and Activate a Virtual Environment** (recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install flask openai spacy nltk requests summarizer keybert flask-cors flask-limiter

4. **Set Your OpenAI API Key**
- MacOS/Linux
   ```bash
   export OPENAI_API_KEY=your_key_here```
- Windows
   ```bash
   set OPENAI_API_KEY=your_key_here```

5. **Download spaCy Model (if not already installed)**
   ```bash
   python -m spacy download en_core_web_sm```

---
## Run ▶️
1. **Start the Flask Sever**
   ```bash
   python app.py```




