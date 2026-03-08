"""
Project 2: Personality Prediction System Through CV Analysis
============================================================
Flask backend — serves index.html and exposes /analyse API endpoint.

Install dependencies:
    pip install flask nltk PyMuPDF

Run:
    python app.py
Then open:  http://localhost:5000
"""

import os
import re
import sys
from collections import Counter

from flask import Flask, request, jsonify, send_from_directory

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

app = Flask(__name__, static_folder=".", template_folder=".")


# ══════════════════════════════════════════════════════════════
# 1.  BIG FIVE KEYWORD DICTIONARIES
# ══════════════════════════════════════════════════════════════

TRAIT_KEYWORDS = {
    "Openness": [
        "creative", "innovative", "research", "design", "art", "music",
        "literature", "explore", "curious", "experiment", "novel", "idea",
        "vision", "imagination", "diverse", "philosophy", "culture",
        "learning", "insight", "discovery", "invention", "strategy",
        "brainstorm", "concept", "aesthetic", "inventor", "artistic",
    ],
    "Conscientiousness": [
        "organized", "detail", "responsible", "plan", "schedule",
        "deadline", "systematic", "accurate", "thorough", "diligent",
        "punctual", "efficient", "reliable", "discipline", "goal",
        "achieve", "complete", "monitor", "quality", "procedure",
        "compliance", "standard", "audit", "structured", "methodical",
        "precise", "meticulous",
    ],
    "Extraversion": [
        "leadership", "team", "communicate", "present", "collaborate",
        "network", "social", "engage", "motivate", "outgoing",
        "negotiation", "coordinate", "facilitate", "mentor", "coach",
        "conference", "public", "speaking", "ambassador", "outreach",
        "stakeholder", "community", "event", "influence", "persuade",
        "spokesperson", "recruit",
    ],
    "Agreeableness": [
        "support", "help", "cooperative", "empathy", "volunteer",
        "assist", "care", "compassion", "patience", "mediate",
        "charity", "welfare", "kind", "trust", "harmony", "counseling",
        "listen", "understanding", "service", "inclusive", "mentor",
        "feedback", "resolve", "reconcile", "considerate", "nurture",
    ],
    "Emotional Stability": [
        "calm", "stable", "consistent", "resilient", "adapt", "stress",
        "pressure", "handle", "manage", "balance", "compose",
        "confident", "assertive", "focused", "steady", "endure",
        "overcome", "professional", "constructive", "objective",
        "rational", "logical", "flexible", "grounded", "persevere",
        "composure",
    ],
}

DESCRIPTIONS = {
    "Openness": {
        "high": "Highly creative and intellectually curious. Thrives in innovative environments.",
        "low":  "Prefers conventional approaches and well-defined tasks.",
    },
    "Conscientiousness": {
        "high": "Extremely organised, goal-oriented, and detail-focused. A reliable performer.",
        "low":  "May prefer flexibility over strict structure.",
    },
    "Extraversion": {
        "high": "Strong communicator and natural leader. Energised by teamwork and networking.",
        "low":  "Works effectively independently; may prefer analytical over social roles.",
    },
    "Agreeableness": {
        "high": "Collaborative, empathetic, and supportive. Excellent team player.",
        "low":  "Direct and task-focused; values results over consensus.",
    },
    "Emotional Stability": {
        "high": "Calm under pressure, resilient, and consistent in performance.",
        "low":  "May be sensitive to change; benefits from structured environments.",
    },
}

TRAIT_EMOJIS = {
    "Openness":            "🎨",
    "Conscientiousness":   "📋",
    "Extraversion":        "🤝",
    "Agreeableness":       "💛",
    "Emotional Stability": "⚖️",
}


# ══════════════════════════════════════════════════════════════
# 2.  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    doc  = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    return file_bytes.decode("utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════
# 3.  NLP PIPELINE
# ══════════════════════════════════════════════════════════════

def preprocess(text: str) -> list:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def score_traits(tokens: list) -> dict:
    freq    = Counter(tokens)
    raw     = {trait: sum(freq.get(kw, 0) for kw in kws)
               for trait, kws in TRAIT_KEYWORDS.items()}
    max_val = max(raw.values()) or 1
    return {t: round((v / max_val) * 100, 1) for t, v in raw.items()}


def interpret(scores: dict) -> dict:
    return {
        trait: DESCRIPTIONS[trait]["high" if score >= 50 else "low"]
        for trait, score in scores.items()
    }


# ══════════════════════════════════════════════════════════════
# 4.  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    if "cv_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file     = request.files["cv_file"]
    name     = request.form.get("candidate_name", "Candidate").strip() or "Candidate"
    filename = file.filename

    if not filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".txt"):
        return jsonify({"error": "Only .pdf and .txt files are supported."}), 400

    try:
        file_bytes = file.read()
        text       = extract_text(file_bytes, filename)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {str(e)}"}), 500

    tokens       = preprocess(text)
    scores       = score_traits(tokens)
    descriptions = interpret(scores)
    dominant     = max(scores, key=scores.get)

    result = {
        "candidate_name": name,
        "dominant_trait": dominant,
        "dominant_emoji": TRAIT_EMOJIS[dominant],
        "dominant_desc":  descriptions[dominant],
        "traits": [
            {
                "name":        trait,
                "emoji":       TRAIT_EMOJIS[trait],
                "score":       scores[trait],
                "level":       "HIGH" if scores[trait] >= 50 else "LOW",
                "description": descriptions[trait],
            }
            for trait in TRAIT_KEYWORDS          # fixed order
        ],
    }
    return jsonify(result)


# ══════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  CV Personality Analyzer running at  http://localhost:5000\n")
    app.run(debug=True, port=5000)