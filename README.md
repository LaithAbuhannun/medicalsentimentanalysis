# 🩺 Medical Sentiment Analysis

Medical Sentiment Analysis is an end-to-end NLP system that classifies clinical / patient-style text into sentiment (for example: **negative**, **neutral**, **positive**).

* Fine-tuned sentiment model trained on medical-style language (not generic tweets/movie reviews).
* Lightweight HTML interface for entering text and seeing predictions instantly.
* Python backend that serves the model for real-time inference.
* Saved artifacts (tokenizer, label encoder, model weights) so results are reproducible.
* Training notebook that documents how the model was built and evaluated.

---

## 🔴 Demo

<p align="center">
  <video src="https://github.com/user-attachments/assets/71154c8a-f665-4bc6-9ee1-4998f2c30713"
         width="480"
         controls
         muted
         playsinline>
  </video>
</p>

**What the demo shows:**

1. User types a medical-style note / message / complaint.
2. Clicks analyze.
3. The app returns a sentiment label.
4. The UI highlights that label so you can quickly read tone.

This is useful for triage (urgent vs calm), patient experience, and QA.

---

## 🧠 Why this matters

Healthcare and clinical communication has its own language.

Examples:

* “Patient denies chest pain post-procedure”
* “Family is extremely upset about wait time”
* “Pain is improving and breathing is stable now”

Generic sentiment models fail on that type of phrasing.

This project specifically:

1. Cleans and normalizes text in a medical context.
2. Runs a fine-tuned sentiment classifier.
3. Returns the sentiment class + confidence so you can quickly understand tone.

Real use cases:

* Flag negative / urgent language in patient messages.
* Monitor satisfaction and frustration trends over time.
* Assist staff in prioritizing outreach.

---

## 🏗 System Overview (What runs where)

```text
[ interface_sentiment.html ]  <-- browser UI for humans
        |
        |  calls the backend with the text you entered
        v
[ server_sentiment/server_sentiment.py ]  <-- Python API service
        |
        |  loads trained model + tokenizer + label encoder
        v
[ server_sentiment/model_artifacts/ ]  <-- all the saved model pieces
    - sentiment_model/            (model config + tokenizer vocab, etc.)
    - label_encoder.pkl           (maps class index -> "negative"/"neutral"/"positive")
```

Plus:

* `NLP_Fine_Tuned_clean.ipynb` documents model training and evaluation.
* `logs.csv` (in `server_sentiment/`) can track requests/results during testing.

So the pipeline is:

**User types text → frontend sends it → backend preprocesses → model predicts → frontend shows label.**

---

## ✨ Core Features

### 🩻 Domain-tuned sentiment model

* The model is fine-tuned specifically for healthcare-style language instead of casual social text.
* This reduces false positives like:

  * “patient denies pain” = actually good/stable, not “negative mood,” etc.

### 🔬 Saved inference artifacts

In `server_sentiment/model_artifacts/` you’ve checked in everything needed for repeatable inference:

* `sentiment_model/`

  * `config.json` – model architecture / config
  * `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` – tokenizer vocab + rules
* `label_encoder.pkl` – converts numeric prediction → readable label (e.g. 0 → "negative")

This proves the model isn’t magic / hardcoded. It’s actually loaded from trained weights.

### 🌐 Live API server

`server_sentiment/server_sentiment.py` runs a backend service (Flask / FastAPI style).
Typical flow:

* Receive text from the client via `POST`.
* Clean/tokenize it using the same tokenizer that was used during training.
* Run inference with the fine-tuned model.
* Return JSON like:

  ```json
  {
    "sentiment": "negative",
    "confidence": 0.91
  }
  ```

This makes your model usable by *other* systems, not just the demo page.

### 🖥 Frontend interface

* `interface_sentiment.html` is the UI for humans.
* It's a simple webpage where you paste or type text.
* You click a button (Analyze / Classify).
* It hits the backend and then shows the predicted sentiment clearly (e.g. colored badge “NEGATIVE” / “POSITIVE”).

This looks like a real product, not just a notebook screenshot.

### 📓 Training notebook

* `NLP_Fine_Tuned_clean.ipynb` (and/or `NLP_Fine_Tuned.ipynb`) is the notebook that:

  * loads/cleans training data,
  * tokenizes it / encodes labels,
  * fine-tunes the model,
  * evaluates accuracy,
  * exports the final artifacts that the server uses.

That notebook = proof of work.

---

## 📂 Repository structure

Here’s what your repo currently looks like based on your layout:

```text
medicalsentimentanalysis/
├─ README.md
├─ interface_sentiment.html          # Frontend UI for entering text + seeing sentiment

├─ NLP_Fine_Tuned_clean.ipynb        # Notebook: data prep, training, evaluation, export
#  (You may also have NLP_Fine_Tuned.ipynb)

├─ server_sentiment/                 # Backend service + model artifacts
│   ├─ server_sentiment.py           # API server for inference
│   ├─ requirements.txt              # All Python deps (Flask/FastAPI, transformers, etc.)
│   ├─ logs.csv                      # Optional log of predictions / test runs
│   │
│   └─ model_artifacts/              # Saved model + preprocessing
│       ├─ label_encoder.pkl         # Converts numeric class -> sentiment label string
│       │
│       └─ sentiment_model/          # Model + tokenizer assets
│           ├─ config.json
│           ├─ tokenizer.json
│           ├─ tokenizer_config.json
│           ├─ special_tokens_map.json
│           ├─ vocab.txt
│           └─ (other model weight files if present)
```

### Why this structure is nice

* `interface_sentiment.html` = presentation layer
* `server_sentiment/` = serving layer
* `model_artifacts/` = ML assets (frozen snapshot of the trained model)
* Notebook at root = “how I built it / prove I trained it”

That separation is exactly what people want to see in applied ML work: train → freeze → serve.

---

## 🔌 API behavior (server_sentiment.py)

Your backend script (`server_sentiment.py`) is responsible for:

1. Loading the tokenizer and model from `model_artifacts/sentiment_model/`.
2. Loading `label_encoder.pkl` to convert numeric prediction to text label.
3. Exposing an endpoint (examples, adjust names if different):

### Example request:

```json
{
  "text": "Patient is extremely upset about the long wait and says pain is worse."
}
```

### Example response:

```json
{
  "sentiment": "negative",
  "confidence": 0.93
}
```

Internally it’s doing:

* preprocess/clean → tokenize → model forward pass → argmax → map class index to label → send back JSON.

This is production-style architecture, not just Jupyter.

---

## 🖥 Frontend flow (interface_sentiment.html)

The HTML interface typically does something like:

1. User types or pastes a note into a text box.
2. JavaScript calls the backend (`fetch('/predict', { text: ... })` style).
3. The response comes back as sentiment + score.
4. The page updates to show:

   * Sentiment label (Negative / Neutral / Positive)
   * Optional probability (confidence)
   * Visual highlight/color for the sentiment

     * red for negative, yellow for neutral, green for positive (for fast triage)

This gives you a “mini dashboard” experience for testing the model interactively.

---

## 📘 Model training (NLP_Fine_Tuned_clean.ipynb)

The notebook documents the full ML lifecycle:

* Dataset loading (medical-like / patient-style text).
* Cleaning text: lowercasing, stripping noise, maybe handling medical phrases.
* Train/val split.
* Fine-tuning a sentiment model (ex: transformer-based classifier).
* Evaluating performance (accuracy, F1, confusion matrix).
* Exporting:

  * tokenizer files (`tokenizer.json`, `vocab.txt`, etc.),
  * model config/weights (`config.json`, etc.),
  * encoders (`label_encoder.pkl`).

Those exports land in `server_sentiment/model_artifacts/`, and that’s what the live API loads.

This shows:

* You actually trained a model,
* You froze the artifacts,
* And you wired it into a running UI.

---

## 🚀 Run it locally

### 1. Install backend dependencies

```bash
cd server_sentiment
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start the inference server

```bash
python server_sentiment.py
```

Now your API should be running locally (for example on `http://localhost:5000` or whatever port is in your code).

### 3. Open the interface

Option A (static HTML):

* Just open `interface_sentiment.html` in your browser.
* Make sure that HTML's JavaScript is pointing to the correct backend URL (like `http://localhost:5000/predict`).

Option B (if the HTML is actually served by the backend):

* Visit the route that serves the interface (for example `/`).
* Type a sentence and test your prediction.

---

## 🛣 Roadmap / future improvements

* [ ] Add more nuanced labels (anxious, urgent, reassured, confused).
* [ ] Highlight the exact phrases that drove the classification (“upset about wait time”).
* [ ] Batch analysis: upload CSV of patient comments and get a report.
* [ ] Dashboard with sentiment trends over time.
* [ ] Containerize / deploy (so teams can call the API securely without running Python manually).

---

## ⚠️ Important notes

* This tool is for **sentiment / tone analysis**, not for clinical diagnosis.
* Do not store or commit actual PHI (personal health information).
* `logs.csv` should not contain real patient identifiers in production.
* The notebook is for demonstration of training and evaluation, not for medical decision-making.

---

## TL;DR for reviewers

**Medical Sentiment Analysis =**

* A fine-tuned healthcare sentiment classifier.
* A backend service that exposes it as an API.
* A simple browser UI to test it live.
* Checked-in model artifacts so results are reproducible.
* A training notebook that proves you actually built / trained the model, not just copied something.

This is end-to-end ML product work: collect → train → ship → interact.
