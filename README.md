# Pune Transit Complaint Analyzer

An academic, local-first prototype that classifies public-transport complaint text into a service category, sentiment, and priority. It is a demonstration of an NLP workflow; it is not an official Pune transit system or decision tool.

## Problem and objectives

Passenger feedback is often written as free text. This prototype demonstrates how a lightweight NLP pipeline can organize that text into three distinct labels: service category, sentiment, and priority. It also compares two baseline classifiers using held-out predictions and provides single and CSV batch analysis.

## Dataset and provenance

`data/sample_complaints.csv` is a small, bundled **synthetic** dataset generated for this project. It contains no real passenger reports and no public dataset was used. The generator uses repeatable illustrative text and simple sentiment/priority label patterns. Synthetic data makes the demo work immediately, but scores on it do not establish performance on actual Pune passenger complaints. The dataset can also be regenerated with `python scripts/generate_sample_data.py`.

| Field | Meaning |
| --- | --- |
| `complaint_text` | Illustrative complaint text (synthetic in the bundled file) |
| `category` | Delay, Overcrowding, Driver/Staff Behaviour, Cleanliness, Route/Stop Issue, Fare/Ticketing, Safety, or Other |
| `sentiment` | Positive, Neutral, or Negative |
| `priority` | Low, Medium, or High; generator label, not a real urgency assessment |

The data helpers check required columns, empty text, exact duplicate rows, unknown labels, and class counts. Batch prediction files need only a `complaint_text` column. Prediction output retains any original columns and adds predicted labels and model-reported probabilities.

## Architecture and workflow

`app.py` provides four Streamlit tabs. `src/data_utils.py` handles the dataset schema and validation; `src/preprocessing.py` normalizes text; `src/modeling.py` builds and selects the classification pipelines; `src/evaluation.py` calculates metrics. Each task has its own model. TF-IDF is fitted within the training partition only. Multinomial Naive Bayes and balanced Logistic Regression are compared. A training-only validation split selects a model by macro-F1; ties prefer Naive Bayes. Final comparisons are calculated on a separate held-out test split. Random seed 42 is used, with stratification when class counts permit and a valid unstratified fallback for small data.

Reported metrics include accuracy, macro and weighted precision/recall/F1, per-class precision/recall/F1/support, and labeled confusion matrices. Macro scores treat classes equally; weighted scores account for class support. The model-reported probabilities are uncalibrated outputs and are not certainty estimates.

## Limitations and responsible use

The synthetic text is repetitive and its labels follow the generator's patterns, which can make results look artificially strong. Real complaints can include Marathi, Hindi, English, code-switching, spelling variation, ambiguity, and context not represented here. A real deployment would need appropriately consented and representative human-reviewed data, privacy protections, language and bias checks, monitoring, and human oversight. Do not use the predictions as official decisions or to determine an individual's treatment. No paid API or external service is used at runtime.

## Local setup

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The application loads the bundled CSV on startup and trains lightweight models locally. A sidebar status panel confirms data and model startup. Streamlit caches the loaded dataset and trained models. The core demo does not depend on a session-local generated asset or persistent filesystem.

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```

Tests cover text and CSV schema validation, duplicate and unknown-label handling, model training and prediction shape, held-out metric output, and fixed-seed reproducibility.

## Streamlit Community Cloud deployment

1. Push this project folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
3. Create a new app, select the repository and branch, and set the app file path to `app.py`.
4. Deploy, wait for startup, then verify the sidebar reports that sample data and models loaded successfully. Try one text prediction and a small CSV upload.

The repository uses a lightweight, bounded `requirements.txt`, case-sensitive paths, and no secrets or machine-specific absolute paths. Streamlit Cloud's local filesystem is ephemeral; the application does not require persistence for its core demo. The Python dependencies are compatible with the recommended Python range; use Python 3.12 on Cloud if selecting a runtime explicitly.

## Example input and output

Example input: `The bus was delayed and I missed my connection`.

Single analysis shows a predicted category, sentiment, and priority, with an optional uncalibrated model-reported probability and an explanation of each field. The exact predictions depend on the trained models and must not be assumed in advance. Batch output adds `predicted_category`, `predicted_sentiment`, `predicted_priority` and corresponding probability columns.
