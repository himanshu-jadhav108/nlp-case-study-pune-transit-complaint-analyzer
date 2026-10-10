# Streamlit Community Cloud Deployment Guide

## 1. Application Overview

**Pune Transit Complaint Analyzer** is an academic NLP triage platform designed for urban public transport feedback. It processes raw passenger feedback text and predicts three target dimensions simultaneously:
- **Service Category** (8 classes: `Delay`, `Overcrowding`, `Driver/Staff Behaviour`, `Cleanliness`, `Route/Stop Issue`, `Fare/Ticketing`, `Safety`, `Other`)
- **Sentiment** (3 classes: `Positive`, `Neutral`, `Negative`)
- **Urgency Priority** (3 classes: `Low`, `Medium`, `High`)

The application features:
- Single-text interactive classification with quick-fill scenario chips and class probability distributions.
- Batch CSV processing with validation, summary KPIs, category/priority filtering, and UTF-8 BOM CSV exports.
- Leakage-conscious model evaluation comparing Multinomial Naive Bayes and Logistic Regression against a held-out test set, complete with confusion matrices and per-class reports.
- Comprehensive data provenance and methodology disclosures with class distribution analytics.

---

## 2. Streamlit Cloud Configuration & Entry Point

| Setting | Value |
| :--- | :--- |
| **Main file path (Entry point)** | `app.py` |
| **Recommended Python Version** | `3.12` (compatible with 3.10–3.14) |
| **App repository root** | Repository root (where `app.py` and `requirements.txt` reside) |
| **Streamlit Config Path** | `.streamlit/config.toml` |
| **Secrets Requirement** | **None** (100% self-contained local ML pipeline) |

---

## 3. Runtime Dependencies

Production dependencies are explicitly specified in [requirements.txt](file:///d:/Case-Study/NLP/Pune_Transit_Complaint_Analyzer/requirements.txt):

```text
streamlit>=1.35.0,<2.0.0
pandas>=2.1.0,<3.0.0
scikit-learn>=1.4.0,<2.0.0
matplotlib>=3.8.0,<4.0.0
numpy>=1.26.0,<3.0.0
```

*Note: For testing and local QA, development tools are listed in `requirements-dev.txt` (`pytest>=8.0.0`).*

---

## 4. Environment Variables and Secrets

- **External Services**: None. The pipeline requires no external database, third-party LLM, or paid cloud API.
- **Secrets Setup**: No secrets are required for standard operation. If extending with third-party webhooks in the future, follow `.streamlit/secrets.toml.example`.

---

## 5. Local Setup and Startup Commands

### Virtual Environment Initialization

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Launching the Application

```powershell
# Windows PowerShell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py

# Linux / macOS
streamlit run app.py
```

### Running Automated QA Tests

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -v
```

---

## 6. Local Validation Summary

Prior to deployment preparation, the following local verifications were performed using the project's dedicated virtual environment (`.\.venv\Scripts\python.exe`):

1. **Interpreter Path**: `D:\Case-Study\NLP\Pune_Transit_Complaint_Analyzer\.venv\Scripts\python.exe`
2. **Pip Integrity (`pip check`)**: **Passed** (`No broken requirements found`).
3. **Streamlit Version**: Verified (`Streamlit 1.65.0`).
4. **Automated Unit Tests**: **10 passed in 2.62s** (`pytest -v`).
5. **Headless Execution Test**: Tested on port 8501 via `Invoke-WebRequest` returning HTTP status **200 OK**.
6. **Path Resolution**: Relative paths verified using `pathlib.Path` with automatic fallback dataset generation.

---

## 7. Streamlit Community Cloud Step-by-Step Deployment

Follow these steps to deploy on Streamlit Community Cloud:

1. **Initialize Git & Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: prepare Pune Transit Complaint Analyzer for Streamlit Cloud"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. **Access Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Create New App**:
   - Click **"Create app"** > **"Deploy an app"**.
   - Select your repository and choose the `main` branch.
4. **Specify Main File Path**:
   - In the **Main file path** input, enter:
     ```
     app.py
     ```
5. **Select Python Version**:
   - Click **Advanced settings**.
   - Under Python version, select **3.12**.
6. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit Cloud will automatically provision a container, install packages from `requirements.txt`, and boot the app.

---

## 8. Runtime & Platform Considerations

- **Ephemeral Storage**: Streamlit Community Cloud instances have ephemeral storage. The application does not rely on disk persistence between sessions; sample data is bundled and cached in-memory via `@st.cache_data` and `@st.cache_resource`.
- **Memory Footprint**: The TF-IDF and scikit-learn models are extremely lightweight (< 200 MB RAM during training and inference), well below Streamlit Cloud's 1 GB free-tier limit.
- **CPU Inference**: Fast inference (< 15 ms per message) without requiring GPU acceleration.
- **Stateless Batch Analysis**: Uploaded files and predictions are processed in memory and delivered immediately via downloadable byte streams without persisting user data on the server.

---

## 9. Troubleshooting Common Deployment Failures

| Issue | Likely Root Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError` on Cloud | Omitted package in `requirements.txt` | Ensure `requirements.txt` is in the repository root and matches the direct dependencies. |
| Streamlit Cloud build timeout | Heavy packages or deep wheel building | The bounded `requirements.txt` installs pre-built binary wheels in under 60 seconds. |
| CSV Upload Encoding Error | Non-standard CSV encoding | The app uses `utf-8-sig` to safely support UTF-8 BOM files from Excel and Windows. |
| App restarts or clears state | Cloud instance reboot | Expected behavior on ephemeral cloud containers; core app data re-initializes cleanly from bundled files. |
