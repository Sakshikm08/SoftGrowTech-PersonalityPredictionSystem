# 🧠 CV Personality Analyzer

Predict Big Five personality traits from any resume using NLP — built with Python & Flask during my internship at **SoftGrowTech**.

## Features
- Upload `.pdf` or `.txt` CV via drag & drop
- Predicts 5 personality traits scored 0–100
- Radar chart, score bars, HIGH/LOW badges
- Personality Type label (e.g. The Innovator)
- Export as `.txt` and print-ready layout

## Tech Stack
Python · Flask · NLTK · PyMuPDF · HTML · CSS · JavaScript · Chart.js

## Setup
```bash
pip install flask nltk PyMuPDF
python app.py
```
Open `http://localhost:5000`

## How It Works
Upload CV → Extract text → NLP pipeline (tokenize, remove stopwords, lemmatize) → Score keywords per trait → Normalize to 0–100 → Display results

## Project Structure
├── app.py        ← Flask server + NLP engine
└── index.html    ← Frontend UI

> Results are indicative only — not a clinical assessment.
