from pathlib import Path
from datetime import datetime, timezone
import csv
import pickle
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "model_artifacts"
MODEL_DIR = ARTIFACTS / "sentiment_model"      # this is your Hugging Face export folder
ENCODER_PATH = ARTIFACTS / "label_encoder.pkl" # saved LabelEncoder from Colab
LOG_PATH = HERE / "logs.csv"

# ── Load model / tokenizer / encoder ────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ── Self-harm keyword list (inline fallback) ────────
SELF_HARM_KEYWORDS = [
     "suicide", "commit suicide", "attempt suicide", "kill myself", "going to kill myself", "want to kill myself", "thinking about killing myself", "off myself", "end it all", "end my life",
     "finish it all", "finish myself off", "hang myself", "overdose", "overdose on pills", "take pills", "swallow pills", "slit my wrists", "slash my arms", "cut myself",
     "self harm", "self-harm", "hurt myself", "want to hurt myself", "cause myself pain", "i'm worthless", "worthless", "not worth living", "life isn't worth living", "no reason to live",
     "what's the point anymore", "feel like dying", "feeling suicidal", "suicidal thoughts", "suicidal ideation", "want it to end", "wish i were dead", "wish i was dead", "hope i'm dead", "death can't come soon enough",
     "ready to die", "can't go on", "can't keep living", "don't want to exist", "don't want to be here", "vanish forever", "disappear forever", "blow my brains out", "shoot myself", "put a bullet in my head",
     "please kill me", "kill me", "just kill me", "drown myself", "suffocate myself", "strangle myself", "crash my car", "jump off a bridge", "jump off a building", "jump in front of a train",
     "walk in front of traffic", "nothing matters", "meaningless life", "pointless existence", "broken beyond repair", "beyond saving", "can't face tomorrow", "everyone would be better off without me", "no one would miss me", "don't deserve to live",
     "wasted life", "time to die", "done with life", "done with this", "i'm done", "my life sucks", "life is meaningless", "final exit", "offing myself", "self destruct", "self destruction", "self electrocute",
     "drown myself","suffocate myself", 'OD', "overdose", "overdose on pills", "take pills", "swallow pills", "slit my wrists", "slash my arms", "cut myself",
     "kill", "die", "hopeless", "useless", "pointless", "empty", "sad", "depressed", "cry", "cut", "hurt", "hate", "disappear", "vanish", "tired", "done", "end", "alone",
     "broken", "slit", "drown", "strangle", "burn", "crash", "numb", "scared", "anxious", "panic", "afraid", "terrified", "angry", "furious", "rage", "mad", "can't go on", "no one cares",
     "nobody cares", "why am i here", "not okay", "life sucks",
]

# ── Flask app setup ─────────────────────────────────
app = Flask(__name__)
CORS(app)


def _analyze_text(text: str):
    """
    Run inference on a single piece of text:
    - tokenize
    - model forward pass (softmax)
    - pick top class
    - map LABEL_x -> human label using label_encoder
    - scan for self-harm terms
    """

    # 1. Tokenize to tensors
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # 2. Model forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits            # shape [1, num_labels]
        probs = F.softmax(logits, dim=1)[0]  # [num_labels]

    # 3. Pick top class
    top_idx = int(torch.argmax(probs).item())
    confidence = float(probs[top_idx].item())

    # Try to decode index -> original sentiment string using your LabelEncoder
    # (If LabelEncoder had classes_ like ["negative","neutral","positive"])
    try:
        decoded_label = label_encoder.inverse_transform([top_idx])[0]
    except Exception:
        # fallback label if encoder somehow mismatches
        decoded_label = f"LABEL_{top_idx}"

    # 4. Self-harm check using simple keyword scan
    lowered = text.lower()
    flagged_terms = [kw for kw in SELF_HARM_KEYWORDS if kw in lowered]
    self_harm_flag = len(flagged_terms) > 0

    return {
        "sentiment": decoded_label,          # e.g. "negative"
        "confidence": confidence,            # float 0-1
        "self_harm_flag": self_harm_flag,    # boolean
        "flagged_terms": flagged_terms       # which phrases matched
    }


def _log_entry(text, result):
    """
    Append the result to logs.csv.
    We'll create logs.csv if it doesn't exist yet.
    """
    newfile = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "text",
                "sentiment",
                "confidence",
                "self_harm_flag"
            ],
        )
        if newfile:
            writer.writeheader()

        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "self_harm_flag": int(result["self_harm_flag"])
        })


def _read_logs():
    """
    Read all prior rows from logs.csv.
    If it's empty, return [].
    """
    if not LOG_PATH.exists():
        return []
    rows = []
    with LOG_PATH.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


@app.post("/analyze")
def analyze_route():
    """
    Request body:
    {
      "text": "patient said ... "
    }

    Response body:
    {
      "sentiment": "negative",
      "confidence": 0.93,
      "self_harm_flag": true,
      "flagged_terms": ["i don't want to live"]
    }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = _analyze_text(text)
    _log_entry(text, result)
    return jsonify(result)


@app.get("/metrics/summary")
def metrics_summary():
    """
    Returns overall metrics for dashboard cards + sentiment chart.
    Shape:
    {
      "total": 12,
      "positive_count": 3,
      "neutral_count": 2,
      "negative_count": 7,
      "negative_rate_percent": "58.3%",
      "self_harm_alerts": 4,
      "latest_risk": "ALERT" | "OK"
    }
    """
    rows = _read_logs()
    if not rows:
        return jsonify({
            "total": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "negative_rate_percent": "0%",
            "self_harm_alerts": 0,
            "latest_risk": "OK"
        })

    total = len(rows)
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    self_harm_alerts = 0

    last = rows[-1]
    latest_risk = "ALERT" if last["self_harm_flag"] == "1" else "OK"

    for row in rows:
        sent = row["sentiment"].lower()
        if "pos" in sent:         # "positive"
            positive_count += 1
        elif "neu" in sent:       # "neutral"
            neutral_count += 1
        else:                     # assume "negative"
            negative_count += 1

        if row["self_harm_flag"] == "1":
            self_harm_alerts += 1

    negative_rate = (negative_count / total * 100.0) if total > 0 else 0.0

    return jsonify({
        "total": total,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "negative_rate_percent": f"{negative_rate:.1f}%",
        "self_harm_alerts": self_harm_alerts,
        "latest_risk": latest_risk
    })


@app.get("/metrics/timeseries")
def metrics_timeseries():
    """
    Time series of self-harm alerts per day.
    Shape:
    [
      { "date": "2025-10-27", "self_harm_alerts": 2 },
      ...
    ]
    """
    rows = _read_logs()
    daily = {}
    for row in rows:
        ts_raw = row["timestamp_utc"]
        if ts_raw.endswith("Z"):
            ts_raw = ts_raw.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(ts_raw)
        except Exception:
            # bad row -> skip
            continue

        day = ts.date().isoformat()
        if day not in daily:
            daily[day] = {"self_harm_alerts": 0}
        if row["self_harm_flag"] == "1":
            daily[day]["self_harm_alerts"] += 1

    # turn dict -> sorted list
    out = []
    for day in sorted(daily.keys()):
        out.append({
            "date": day,
            "self_harm_alerts": daily[day]["self_harm_alerts"]
        })
    return jsonify(out)


@app.get("/sessions/recent")
def sessions_recent():
    """
    Recent individual analyses (for the table).
    Shape:
    [
      {
        "timestamp_utc": "...",
        "text_snippet": "...",
        "sentiment": "negative",
        "confidence": 0.93,
        "self_harm_flag": true
      },
      ...
    ]
    """
    rows = _read_logs()
    rows = rows[-20:]  # last 20 entries

    out = []
    for row in rows[::-1]:  # newest first
        snippet = row["text"]
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."

        out.append({
            "timestamp_utc": row["timestamp_utc"],
            "text_snippet": snippet,
            "sentiment": row["sentiment"],
            "confidence": float(row["confidence"]),
            "self_harm_flag": (row["self_harm_flag"] == "1")
        })

    return jsonify(out)


if __name__ == "__main__":
    # Run locally: activate venv, then:
    # python server_sentiment.py
    app.run(host="127.0.0.1", port=5001, debug=True)
