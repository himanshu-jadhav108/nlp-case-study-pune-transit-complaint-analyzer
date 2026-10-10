"""Leakage-conscious model selection and prediction for three label tasks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data_utils import CATEGORY_COLUMN, PRIORITY_COLUMN, SENTIMENT_COLUMN, TEXT_COLUMN
from .preprocessing import normalize_text

TASKS = {"Category": CATEGORY_COLUMN, "Sentiment": SENTIMENT_COLUMN, "Priority": PRIORITY_COLUMN}
SEED = 42


def _split_indices(y: pd.Series, seed: int = SEED):
    indices = np.arange(len(y))
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    try:
        return train_test_split(indices, test_size=0.25, random_state=seed, stratify=stratify), ("stratified" if stratify is not None else "unstratified fallback")
    except ValueError:
        return train_test_split(indices, test_size=0.25, random_state=seed), "unstratified fallback"


def make_pipeline(model_name: str) -> Pipeline:
    classifier = MultinomialNB(alpha=0.5) if model_name == "Multinomial Naive Bayes" else LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    return Pipeline([("tfidf", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), min_df=1, sublinear_tf=True)), ("classifier", classifier)])


def train_models(data: pd.DataFrame) -> dict:
    """Select by validation macro F1 on the training partition, then evaluate test once."""
    if len(data) < 8:
        raise ValueError("At least 8 labeled rows are required to create training, validation, and held-out test partitions.")
    outputs = {"models": {}, "evaluations": {}, "comparisons": {}, "split": {}, "selection": {}, "model_names": ["Multinomial Naive Bayes", "Logistic Regression"]}
    for task, target in TASKS.items():
        y = data[target].astype(str).reset_index(drop=True)
        x = data[TEXT_COLUMN].astype(str).map(normalize_text).reset_index(drop=True)
        (trainval, test), split_kind = _split_indices(y)
        local_y = y.iloc[trainval].reset_index(drop=True)
        # Validation split comes only from training rows; stratify when every class allows it.
        val_stratify = local_y if local_y.value_counts().min() >= 2 else None
        try:
            fit_i, val_i = train_test_split(np.arange(len(trainval)), test_size=0.25, random_state=SEED, stratify=val_stratify)
        except ValueError:
            fit_i, val_i = train_test_split(np.arange(len(trainval)), test_size=0.25, random_state=SEED)
        scores = {}
        fitted = {}
        for name in outputs["model_names"]:
            candidate = make_pipeline(name)
            candidate.fit(x.iloc[trainval].iloc[fit_i], local_y.iloc[fit_i])
            scores[name] = float(f1_score(local_y.iloc[val_i], candidate.predict(x.iloc[trainval].iloc[val_i]), average="macro", zero_division=0))
            fitted[name] = make_pipeline(name).fit(x.iloc[trainval], y.iloc[trainval])
        selected = max(outputs["model_names"], key=lambda n: (scores[n], -outputs["model_names"].index(n)))
        y_true = y.iloc[test].to_numpy()
        outputs["comparisons"][task] = {}
        for name in outputs["model_names"]:
            outputs["comparisons"][task][name] = {"model": fitted[name], "y_true": y_true,
                "y_pred": fitted[name].predict(x.iloc[test]), "test_texts": x.iloc[test].to_numpy(),
                "labels": sorted(y.unique()), "validation_macro_f1": scores[name]}
        model = fitted[selected]
        y_pred = model.predict(x.iloc[test])
        outputs["models"][task] = model
        outputs["evaluations"][task] = {"y_true": y_true, "y_pred": y_pred, "test_texts": x.iloc[test].to_numpy(), "labels": sorted(y.unique())}
        outputs["split"][task] = {"kind": split_kind, "train_rows": int(len(trainval)), "test_rows": int(len(test))}
        outputs["selection"][task] = {"selected": selected, "validation_macro_f1": scores}
    return outputs


def predict_one(bundle: dict, complaint: str) -> dict:
    text = normalize_text(complaint)
    result = {}
    for task, model in bundle["models"].items():
        predicted = str(model.predict([text])[0])
        classifier = model.named_steps["classifier"]
        probabilities = model.predict_proba([text])[0] if hasattr(classifier, "predict_proba") else None
        confidence = float(np.max(probabilities)) if probabilities is not None else None
        prob_dict = {}
        if probabilities is not None and hasattr(classifier, "classes_"):
            prob_dict = {str(cls): float(prob) for cls, prob in zip(classifier.classes_, probabilities)}
        result[task] = {"prediction": predicted, "confidence": confidence, "probabilities": prob_dict}
    return result

