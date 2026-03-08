"""
Project 2: Personality Prediction System Through CV Analysis
============================================================
Predicts Big Five personality traits from a resume/CV file.

Install dependencies (run once):
    pip install nltk PyMuPDF scikit-learn
"""

import os
import re
import sys
from collections import Counter

import nltk
nltk.download("stopwords",  quiet=True)
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("wordnet",    quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer


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


# ══════════════════════════════════════════════════════════════
# 2.  INTERACTIVE INPUT  (no hardcoded paths)
# ══════════════════════════════════════════════════════════════

def prompt_file_path() -> str:
    """Ask the user for a CV file path and validate it."""
    while True:
        path = input("\n  Enter path to CV file (.pdf or .txt): ").strip().strip('"').strip("'")
        if not path:
            print("  [!] No path entered. Please try again.")
            continue
        if not os.path.isfile(path):
            print(f"  [!] File not found: {path}")
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".pdf", ".txt"):
            print("  [!] Unsupported format. Please provide a .pdf or .txt file.")
            continue
        return path


def prompt_candidate_name() -> str:
    """Ask the user for the candidate's name (optional)."""
    name = input("  Enter candidate name (press Enter to skip): ").strip()
    return name if name else "Candidate"


def prompt_save_report():
    """Ask whether to save the report and where."""
    choice = input("  Save report to a .txt file? (y/n): ").strip().lower()
    if choice == "y":
        out = input("  Output path (e.g. report.txt): ").strip().strip('"').strip("'")
        return out if out else "personality_report.txt"
    return None


# ══════════════════════════════════════════════════════════════
# 3.  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        sys.exit("  [!] PyMuPDF not installed. Run:  pip install PyMuPDF")
    doc  = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_cv_text(file_path: str) -> str:
    """Load CV content from a .pdf or .txt file."""
    if os.path.splitext(file_path)[1].lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════
# 4.  PRE-PROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess(text: str) -> list:
    """Lowercase → strip non-alpha → tokenise → remove stopwords → lemmatise."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


# ══════════════════════════════════════════════════════════════
# 5.  TRAIT SCORING
# ══════════════════════════════════════════════════════════════

def score_traits(tokens: list) -> dict:
    """Count keyword hits per trait and normalise to 0-100."""
    freq    = Counter(tokens)
    raw     = {trait: sum(freq.get(kw, 0) for kw in kws)
               for trait, kws in TRAIT_KEYWORDS.items()}
    max_val = max(raw.values()) or 1
    return {t: round((v / max_val) * 100, 1) for t, v in raw.items()}


# ══════════════════════════════════════════════════════════════
# 6.  INTERPRETATION
# ══════════════════════════════════════════════════════════════

def interpret(scores: dict) -> dict:
    return {
        trait: DESCRIPTIONS[trait]["high" if score >= 50 else "low"]
        for trait, score in scores.items()
    }


def dominant_trait(scores: dict) -> str:
    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════════
# 7.  PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_report(scores: dict, descriptions: dict, name: str = "Candidate") -> None:
    BAR  = 30
    LINE = "=" * 64

    print(f"\n{LINE}")
    print(f"   PERSONALITY PREDICTION REPORT  —  {name.upper()}")
    print(LINE)

    for trait in TRAIT_KEYWORDS:          # guaranteed order
        score  = scores.get(trait, 0.0)
        desc   = descriptions.get(trait, "")
        filled = int((score / 100) * BAR)
        bar    = "█" * filled + "░" * (BAR - filled)
        level  = "HIGH" if score >= 50 else "LOW "
        print(f"\n  {trait:<22}  {score:>5}%  [{bar}]  {level}")
        print(f"  └─ {desc}")

    print(f"\n{LINE}")
    print(f"   Dominant Trait : {dominant_trait(scores)}")
    print(LINE + "\n")


# ══════════════════════════════════════════════════════════════
# 8.  SAVE REPORT
# ══════════════════════════════════════════════════════════════

def save_report(scores: dict, descriptions: dict, name: str, output_path: str) -> None:
    BAR  = 30
    lines = [
        f"PERSONALITY PREDICTION REPORT — {name.upper()}",
        "=" * 64, "",
    ]
    for trait in TRAIT_KEYWORDS:
        score  = scores.get(trait, 0.0)
        desc   = descriptions.get(trait, "")
        filled = int((score / 100) * BAR)
        bar    = "█" * filled + "░" * (BAR - filled)
        level  = "HIGH" if score >= 50 else "LOW"
        lines.append(f"{trait:<22}  {score:>5}%  [{bar}]  {level}")
        lines.append(f"  └─ {desc}")
        lines.append("")
    lines.append(f"Dominant Trait: {dominant_trait(scores)}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✔  Report saved → {output_path}\n")


# ══════════════════════════════════════════════════════════════
# 9.  FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def analyse_cv_file(file_path: str, candidate_name: str = "Candidate",
                    save_to=None) -> dict:
    """Load file → preprocess → score → print → optionally save."""
    print(f"\n  Reading: {file_path} …")
    cv_text      = load_cv_text(file_path)
    tokens       = preprocess(cv_text)
    scores       = score_traits(tokens)
    descriptions = interpret(scores)
    print_report(scores, descriptions, name=candidate_name)
    if save_to:
        save_report(scores, descriptions, candidate_name, save_to)
    return scores


# ══════════════════════════════════════════════════════════════
# 10.  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("   CV PERSONALITY ANALYZER  —  Big Five Trait Predictor")
    print("=" * 64)

    file_path      = prompt_file_path()
    candidate_name = prompt_candidate_name()
    save_path      = prompt_save_report()

    analyse_cv_file(file_path, candidate_name, save_to=save_path)