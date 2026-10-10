from __future__ import annotations

from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_utils import (
    CATEGORIES,
    PRIORITIES,
    REQUIRED_COLUMNS,
    SENTIMENTS,
    TEXT_COLUMN,
    build_synthetic_dataset,
    prepare_batch_frame,
    read_uploaded_csv,
    validate_dataset,
)
from src.evaluation import metric_summary
from src.modeling import TASKS, predict_one, train_models

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_complaints.csv"

st.set_page_config(
    page_title="Pune Transit Complaint Analyzer",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom scoped styles for visual polish and accessibility
st.markdown(
    """
    <style>
    /* Design Tokens */
    :root {
        --color-primary: #0284c7;
        --color-primary-dark: #0369a1;
        --color-surface: #ffffff;
        --color-bg-subtle: #f8fafc;
        --color-border: #e2e8f0;
        --color-text-main: #0f172a;
        --color-text-muted: #64748b;
        --color-success: #10b981;
        --color-warning: #f59e0b;
        --color-danger: #ef4444;
    }



    /* Result Dossier Cards */
    .dossier-card {
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .dossier-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .dossier-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--color-text-muted);
        margin-bottom: 0.35rem;
    }
    .dossier-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--color-text-main);
        margin-bottom: 0.5rem;
    }
    .dossier-desc {
        font-size: 0.82rem;
        color: var(--color-text-muted);
        line-height: 1.35;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-positive { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-neutral { background-color: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
    .badge-negative { background-color: #ffe4e6; color: #9f1239; border: 1px solid #fecdd3; }
    .badge-high { background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-medium { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-low { background-color: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }

    /* Telemetry Sidebar Card */
    .sidebar-telemetry {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
    }
    .telemetry-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
        color: #475569;
    }
    .telemetry-item strong {
        color: #0f172a;
    }

    /* Clean Streamlit elements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading demonstration data…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else build_synthetic_dataset()


@st.cache_resource(show_spinner="Training and evaluating lightweight models…")
def load_bundle(data: pd.DataFrame) -> dict:
    return train_models(data)


# Load data and validate
data = load_data()
valid_data, validation_notes, validation_counts = validate_dataset(data)

if len(valid_data) < 12:
    st.error("The bundled sample dataset is missing or too small to train demonstration models.")
    st.stop()

bundle = load_bundle(valid_data)

# Application Header
st.title("🚌 Pune Transit Complaint Analyzer")
st.markdown(
    """
    <div style="display: inline-block; background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; border-radius: 9999px; padding: 3px 12px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;">
        PUNE URBAN MOBILITY · NLP INTELLIGENCE LAB
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Local-first NLP classification engine for transit feedback. Concurrently predicts "
    "**Service Category** (8 classes), **Sentiment** (3 classes), and **Urgency Priority** (3 classes) "
    "using held-out validated models."
)


# Sidebar System Telemetry & Guidance
with st.sidebar:
    st.markdown("### 🚦 System Status")
    st.markdown(
        f"""
        <div class="sidebar-telemetry">
            <div class="telemetry-item"><span>Dataset:</span><strong>{len(valid_data)} Synthetic Rows</strong></div>
            <div class="telemetry-item"><span>Active Pipelines:</span><strong>3 Classifiers</strong></div>
            <div class="telemetry-item"><span>Algorithms:</span><strong>TF-IDF + MNB / LogReg</strong></div>
            <div class="telemetry-item"><span>Privacy Mode:</span><strong>100% Local / Zero API</strong></div>
            <div class="telemetry-item"><span>Runtime Host:</span><strong>Streamlit Cloud Ready</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.success("✅ Models & vectorizers initialized")

    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown(
            """
            This academic prototype demonstrates reproducible NLP triage for public bus transport.
            
            - **No External Network Calls**: Runs self-contained in Python.
            - **Leakage-Safe**: Model selection uses training-only validation splits before held-out testing.
            - **Synthetic Provenance**: Synthetic data illustrates methodology; not official transit dispatch.
            """
        )

    with st.expander("🛠️ Quick Testing Tips", expanded=False):
        st.markdown(
            """
            - Try the **quick-fill prompt chips** in the single analysis tab.
            - Use the **bundled 10-row batch sample** in the batch tab for one-click testing.
            - Inspect the **confusion matrices and per-class reports** in model evaluation.
            """
        )

    st.caption("Pune Transit NLP Prototype · Local & Cloud Deployment")


# Main Tabs Navigation
tab_analyze, tab_batch, tab_eval, tab_method = st.tabs(
    [
        "🔍 Analyze Complaint",
        "📂 Batch Analysis",
        "📊 Model Evaluation",
        "📖 Dataset & Methodology",
    ]
)

# Category icon helper
CATEGORY_ICONS = {
    "Delay": "⏱️",
    "Overcrowding": "👥",
    "Driver/Staff Behaviour": "🪪",
    "Cleanliness": "🧹",
    "Route/Stop Issue": "📍",
    "Fare/Ticketing": "🎟️",
    "Safety": "⚠️",
    "Other": "📌",
}

# -------------------------------------------------------------
# TAB 1: Single Complaint Analysis
# -------------------------------------------------------------
with tab_analyze:
    st.markdown("### Single Complaint Classification")
    st.markdown(
        "Enter a public transit service message below or pick one of the sample scenarios to test the classifier."
    )

    # Sample prompt quick-fill chips
    st.markdown("**Quick-fill sample scenarios:**")
    prompt_cols = st.columns(5)
    sample_scenarios = [
        ("⏱️ Swargate Delay", "The bus was delayed by 35 minutes at Swargate and I missed my connection"),
        ("👥 Katraj Overcrowding", "Too many people were standing in the crowded bus and passengers could not board"),
        ("⚠️ Speeding Hazard", "The bus was driven dangerously fast through Pune station traffic and doors were open"),
        ("🧹 Vehicle Cleanliness", "The bus floor was dirty, seats had rubbish, and vehicle smelled bad"),
        ("🌟 Courteous Conductor", "The conductor kindly helped an elderly passenger find the correct stop"),
    ]

    for col, (label, prompt_text) in zip(prompt_cols, sample_scenarios):
        if col.button(label, use_container_width=True):
            st.session_state["complaint_input"] = prompt_text

    initial_text = st.session_state.get(
        "complaint_input", "The bus was delayed by 30 minutes and I missed my connection"
    )

    with st.form("single_complaint_form"):
        text = st.text_area(
            "Complaint Text",
            value=initial_text,
            height=110,
            placeholder="Type or paste the passenger feedback message…",
            help="Enter English passenger feedback text. Marathi/Hindi support is documented in Methodology.",
        )
        submitted = st.form_submit_button("Run Classification Analysis", type="primary", use_container_width=True)

    if submitted or "last_prediction" in st.session_state:
        if submitted:
            st.session_state["last_text"] = text
            st.session_state["last_prediction"] = predict_one(bundle, text) if text.strip() else None

        active_text = st.session_state.get("last_text", text)
        predictions = st.session_state.get("last_prediction")

        if not active_text.strip():
            st.error("⚠️ Please enter non-empty complaint text before analyzing.")
        elif predictions:
            st.markdown("#### 🎯 Classification Dossier")

            col_cat, col_sent, col_prio = st.columns(3)

            # Category Card
            cat_res = predictions["Category"]
            cat_name = cat_res["prediction"]
            cat_icon = CATEGORY_ICONS.get(cat_name, "📌")
            cat_conf = cat_res["confidence"]
            with col_cat:
                st.markdown(
                    f"""
                    <div class="dossier-card">
                        <div class="dossier-label">Service Domain</div>
                        <div class="dossier-value">{cat_icon} {cat_name}</div>
                        <div class="dossier-desc">Primary service category identified from vocabulary patterns.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if cat_conf is not None:
                    st.progress(min(max(cat_conf, 0.0), 1.0))
                    st.caption(f"Confidence score: **{cat_conf:.1%}** (uncalibrated probability)")

            # Sentiment Card
            sent_res = predictions["Sentiment"]
            sent_name = sent_res["prediction"]
            sent_conf = sent_res["confidence"]
            badge_class = f"badge-{sent_name.lower()}"
            with col_sent:
                st.markdown(
                    f"""
                    <div class="dossier-card">
                        <div class="dossier-label">Emotional Tone</div>
                        <div class="dossier-value">
                            {sent_name}
                            <span class="badge {badge_class}">{sent_name}</span>
                        </div>
                        <div class="dossier-desc">Overall emotional polarity inferred from message language.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if sent_conf is not None:
                    st.progress(min(max(sent_conf, 0.0), 1.0))
                    st.caption(f"Confidence score: **{sent_conf:.1%}** (uncalibrated probability)")

            # Priority Card
            prio_res = predictions["Priority"]
            prio_name = prio_res["prediction"]
            prio_conf = prio_res["confidence"]
            prio_badge = f"badge-{prio_name.lower()}"
            with col_prio:
                st.markdown(
                    f"""
                    <div class="dossier-card">
                        <div class="dossier-label">Urgency Priority</div>
                        <div class="dossier-value">
                            {prio_name}
                            <span class="badge {prio_badge}">{prio_name}</span>
                        </div>
                        <div class="dossier-desc">Demonstration urgency grade learned from synthetic patterns.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if prio_conf is not None:
                    st.progress(min(max(prio_conf, 0.0), 1.0))
                    st.caption(f"Confidence score: **{prio_conf:.1%}** (uncalibrated probability)")

            # Detailed Probability Distribution Breakdown
            with st.expander("📊 Inspect Full Class Probability Distributions", expanded=False):
                prob_tabs = st.tabs(["Category Probabilities", "Sentiment Probabilities", "Priority Probabilities"])
                for tab, task_name in zip(prob_tabs, ["Category", "Sentiment", "Priority"]):
                    with tab:
                        task_probs = predictions[task_name].get("probabilities", {})
                        if task_probs:
                            prob_df = (
                                pd.DataFrame(list(task_probs.items()), columns=["Class", "Probability"])
                                .sort_values(by="Probability", ascending=False)
                                .reset_index(drop=True)
                            )
                            c1, c2 = st.columns([3, 2])
                            with c1:
                                st.bar_chart(prob_df.set_index("Class")["Probability"], height=220)
                            with c2:
                                st.dataframe(
                                    prob_df.style.format({"Probability": "{:.2%}"}),
                                    width="stretch",
                                    hide_index=True,
                                )

                        else:
                            st.info("Probability distribution unavailable for this classifier.")

            st.info(
                "💡 **Interpreting Results:** These three predictions are generated by independent TF-IDF models "
                "trained on the synthetic demonstration corpus. Probabilities indicate internal model softmax/multinomial "
                "scores and are uncalibrated estimates."
            )

# -------------------------------------------------------------
# TAB 2: Batch CSV Analysis
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch CSV Analysis")
    st.markdown(
        "Upload a CSV file containing a `complaint_text` column to perform automated multi-task classification across all rows."
    )

    action_c1, action_c2 = st.columns([1, 1])

    # Template generator
    template_df = pd.DataFrame(
        {
            "complaint_text": [
                "The bus was delayed by 25 minutes and did not arrive on time",
                "The driver was polite and helped passengers board comfortably",
                "Bus was overcrowded and doors did not close safely",
                "Ticket machine charged me twice at the counter",
            ],
            "route_number": ["11", "24", "42", "108"],
            "stop_name": ["Swargate", "Shivajinagar", "Katraj", "Deccan Gymkhana"],
        }
    )
    template_csv = template_df.to_csv(index=False).encode("utf-8-sig")

    with action_c1:
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=template_csv,
            file_name="pune_transit_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a pre-formatted CSV template with valid complaint_text column.",
        )

    with action_c2:
        load_bundled_test = st.button(
            "⚡ Load Bundled 10-Row Test Batch",
            use_container_width=True,
            help="Quickly evaluate batch classification without uploading an external file.",
        )

    uploaded = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="batch_csv_uploader",
        help="Must contain a 'complaint_text' column. Any additional columns are preserved in output.",
    )

    batch_bytes = None
    if uploaded is not None:
        batch_bytes = uploaded.getvalue()
    elif load_bundled_test:
        sample_slice = valid_data.head(10)[[TEXT_COLUMN]].copy()
        sample_slice["route_id"] = [f"R-{i+1}" for i in range(len(sample_slice))]
        batch_bytes = sample_slice.to_csv(index=False).encode("utf-8-sig")

    if batch_bytes is not None:
        try:
            batch = read_uploaded_csv(batch_bytes)
            clean, skipped = prepare_batch_frame(batch)

            if skipped:
                st.warning(f"⚠️ Skipped {skipped} row(s) containing missing or empty complaint text.")

            if clean.empty:
                st.error("No valid non-empty complaint text rows found in uploaded dataset.")
            else:
                with st.spinner(f"Analyzing {len(clean)} rows across Category, Sentiment, and Priority…"):
                    records = [predict_one(bundle, t) for t in clean[TEXT_COLUMN].astype(str)]
                    for task, column in [
                        ("Category", "predicted_category"),
                        ("Sentiment", "predicted_sentiment"),
                        ("Priority", "predicted_priority"),
                    ]:
                        clean[column] = [r[task]["prediction"] for r in records]
                        clean[column + "_probability"] = [r[task]["confidence"] for r in records]

                st.success(f"✅ Successfully processed {len(clean)} complaint record(s).")

                # Summary KPIs
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                top_cat = clean["predicted_category"].mode().iloc[0] if not clean.empty else "N/A"
                neg_ratio = (clean["predicted_sentiment"] == "Negative").mean() if not clean.empty else 0.0
                high_count = int((clean["predicted_priority"] == "High").sum()) if not clean.empty else 0

                kpi1.metric("Processed Rows", len(clean))
                kpi2.metric("Dominant Category", top_cat)
                kpi3.metric("Negative Sentiment", f"{neg_ratio:.1%}")
                kpi4.metric("High Priority Triage", f"{high_count} rows")

                # Interactive Filtering
                filter_c1, filter_c2 = st.columns(2)
                with filter_c1:
                    cat_filter = st.multiselect(
                        "Filter by Predicted Category",
                        options=sorted(clean["predicted_category"].unique()),
                        default=sorted(clean["predicted_category"].unique()),
                    )
                with filter_c2:
                    prio_filter = st.multiselect(
                        "Filter by Predicted Priority",
                        options=sorted(clean["predicted_priority"].unique()),
                        default=sorted(clean["predicted_priority"].unique()),
                    )

                filtered_df = clean[
                    clean["predicted_category"].isin(cat_filter)
                    & clean["predicted_priority"].isin(prio_filter)
                ]

                st.dataframe(
                    filtered_df,
                    width="stretch",
                    hide_index=True,
                )


                # Export
                csv_export = filtered_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Download Filtered Predictions as CSV",
                    data=csv_export,
                    file_name="pune_transit_predictions.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"❌ Error processing CSV. Please ensure valid UTF-8 encoding. Details: {exc}")

# -------------------------------------------------------------
# TAB 3: Model Evaluation
# -------------------------------------------------------------
with tab_eval:
    st.markdown("### Held-Out Model Evaluation & Comparison")
    st.markdown(
        "Each task is evaluated on a held-out test split (25%). Model selection between "
        "**Multinomial Naive Bayes** and **Logistic Regression** is determined strictly using "
        "macro-F1 on the training-set validation partition to prevent evaluation leakage."
    )

    selected_task = st.selectbox("Select Classification Task", list(TASKS.keys()), index=0)

    # Comparison summary
    rows = []
    for name, item in bundle["comparisons"][selected_task].items():
        metrics = metric_summary(item["y_true"], item["y_pred"], item["labels"])
        is_selected = name == bundle["selection"][selected_task]["selected"]
        rows.append(
            {
                "Model": name,
                "Status": "Selected 🏆" if is_selected else "Candidate",
                "Val Macro F1 (Selection)": item["validation_macro_f1"],
                "Test Accuracy": metrics["accuracy"],
                "Test Macro Precision": metrics["macro_precision"],
                "Test Macro Recall": metrics["macro_recall"],
                "Test Macro F1": metrics["macro_f1"],
                "Test Weighted F1": metrics["weighted_f1"],
            }
        )

    cmp_df = pd.DataFrame(rows)
    st.dataframe(
        cmp_df.style.format(
            {
                "Val Macro F1 (Selection)": "{:.3f}",
                "Test Accuracy": "{:.3f}",
                "Test Macro Precision": "{:.3f}",
                "Test Macro Recall": "{:.3f}",
                "Test Macro F1": "{:.3f}",
                "Test Weighted F1": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


    selected_model_name = bundle["selection"][selected_task]["selected"]
    split_info = bundle["split"][selected_task]
    st.caption(
        f"🏆 **Selected Model:** {selected_model_name} (chosen by higher validation macro-F1; ties prefer Naive Bayes). "
        f"Held-out test set size: {split_info['test_rows']} rows ({split_info['kind']})."
    )

    st.markdown("---")
    st.markdown(f"#### Detailed Performance: {selected_task}")

    inspect_model = st.selectbox(
        "Select Model for Detailed Confusion Matrix and Classification Report",
        bundle["model_names"],
        index=bundle["model_names"].index(selected_model_name),
    )

    chosen = bundle["comparisons"][selected_task][inspect_model]
    metrics = metric_summary(chosen["y_true"], chosen["y_pred"], chosen["labels"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
    m3.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")
    m4.metric("Macro F1-Score", f"{metrics['macro_f1']:.3f}")

    col_matrix, col_rep = st.columns([1, 1])

    with col_matrix:
        st.markdown("**Held-Out Confusion Matrix**")
        labels = chosen["labels"]
        matrix = metrics["confusion_matrix"]

        # Styled Matplotlib Figure
        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

        ax.set_title(f"{selected_task}: {inspect_model}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=9, labelpad=8)
        ax.set_ylabel("True Ground Truth Label", fontsize=9, labelpad=8)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_rep:
        st.markdown("**Per-Class Precision, Recall & F1**")
        report = pd.DataFrame(metrics["report"]).T
        valid_indices = [x for x in chosen["labels"] if x in report.index] + ["macro avg", "weighted avg"]
        filtered_report = report.loc[valid_indices][["precision", "recall", "f1-score", "support"]]
        st.dataframe(
            filtered_report.style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}
            ),
            width="stretch",
        )
        st.caption(
            "Macro avg weights each class equally regardless of support. Weighted avg scales by number of true instances."
        )

# -------------------------------------------------------------
# TAB 4: Dataset & Methodology
# -------------------------------------------------------------
with tab_method:
    st.markdown("### Dataset Provenance & Engineering Methodology")

    st.warning(
        "⚠️ **Synthetic Demonstration Dataset Notice:** The bundled dataset (`data/sample_complaints.csv`) "
        "is deterministically generated to demonstrate the NLP pipeline. It contains zero real passenger submissions "
        "and does not reflect actual Pune Mahanagar Parivahan Mahamandal Limited (PMPML) performance."
    )

    col_data_prev, col_data_dist = st.columns([1, 1])

    with col_data_prev:
        st.markdown("#### Sample Records (Synthetic)")
        st.dataframe(data.head(10), width="stretch", hide_index=True)

        st.caption(f"Total bundled records: {len(data)} rows across 8 transit categories.")

    with col_data_dist:
        st.markdown("#### Class Distributions Across Tasks")
        dist_task = st.radio("Select distribution to visualize:", ["Category", "Sentiment", "Priority"], horizontal=True)
        target_col = TASKS[dist_task]
        counts = data[target_col].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        st.bar_chart(counts.set_index("Class")["Count"], height=240)

    st.markdown("---")
    st.markdown("#### NLP Pipeline Architecture")

    st.markdown(
        """
        ```mermaid
        graph TD
            A[Raw Passenger Text] --> B[Text Normalization<br/>Lowercase + Whitespace Cleaning]
            B --> C[TF-IDF Vectorizer<br/>Unigrams + Bigrams, Sublinear TF]
            C --> D1[Category Classifier<br/>8 Classes]
            C --> D2[Sentiment Classifier<br/>Positive / Neutral / Negative]
            C --> D3[Priority Classifier<br/>Low / Medium / High]
            D1 --> E[Consolidated Triage Dossier]
            D2 --> E
            D3 --> E
        ```
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Technical Implementation Standards")
    st.markdown(
        """
        1. **Data Leakage Prevention:** The `TfidfVectorizer` is fitted exclusively on the training partition. 
           Validation and held-out test folds are transformed using the fitted vocabulary.
        2. **Reproducible Stratified Partitioning:** Splits use `random_state=42` with stratified sampling 
           across all label distributions, falling back gracefully to unstratified splits if class counts are minimal.
        3. **Dual Model Benchmarking:** Compares `MultinomialNB(alpha=0.5)` with `LogisticRegression(class_weight='balanced')`.
           Selection is strictly guided by validation macro-F1 to optimize balanced representation.
        4. **Pune Multilingual Context & Production Roadmap:**
           - *Language Diversity:* Pune commuters communicate in Marathi (मराठी), Hindi (हिंदी), English, and colloquial Romanized Marathi (e.g., 'bus swargate la late aali').
           - *Future Roadmap:* A production deployment requires multilingual transformer embeddings (such as IndicBERT or multilingual MiniLM), human review in the loop, anonymization of passenger PII, and integration with transit dispatch ticketing APIs.
        """
    )

    with st.expander("🔍 Dataset Quality Audit & Schema Validation", expanded=False):
        st.write(
            f"**Total Input Rows:** {validation_counts['input_rows']} · "
            f"**Valid Rows:** {validation_counts['valid_rows']} · "
            f"**Empty Text Exclusions:** {validation_counts['empty_text_rows']} · "
            f"**Duplicate Removals:** {validation_counts['duplicate_rows']} · "
            f"**Unknown Label Rejections:** {validation_counts['unknown_label_rows']}"
        )
        st.caption("Validated against strict schema: " + ", ".join(REQUIRED_COLUMNS))
