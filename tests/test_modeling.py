import numpy as np
import pandas as pd
import pytest

from src.data_utils import build_synthetic_dataset
from src.evaluation import metric_summary
from src.modeling import predict_one, train_models


def test_model_fitting_and_prediction_shape():
    bundle = train_models(build_synthetic_dataset())
    result = predict_one(bundle, "The bus was delayed and the ticket machine failed")
    assert set(result) == {"Category", "Sentiment", "Priority"}
    assert all("prediction" in item for item in result.values())


def test_metrics_and_confusion_matrix_are_from_predictions():
    summary = metric_summary(["a", "b", "a"], ["a", "a", "a"], ["a", "b"])
    assert summary["accuracy"] == 2 / 3
    assert summary["confusion_matrix"].shape == (2, 2)
    assert "macro avg" in summary["report"]


def test_fixed_seed_produces_reproducible_split_and_predictions():
    data = build_synthetic_dataset()
    first, second = train_models(data), train_models(data)
    for task in first["evaluations"]:
        np.testing.assert_array_equal(first["evaluations"][task]["y_true"], second["evaluations"][task]["y_true"])
        np.testing.assert_array_equal(first["evaluations"][task]["y_pred"], second["evaluations"][task]["y_pred"])


def test_too_small_training_data_has_clear_validation_error():
    with pytest.raises(ValueError, match="At least 8 labeled rows"):
        train_models(build_synthetic_dataset().iloc[:5])


def test_prediction_probabilities_structure():
    bundle = train_models(build_synthetic_dataset())
    result = predict_one(bundle, "Bus was 30 mins late at Swargate")
    for task in ["Category", "Sentiment", "Priority"]:
        probs = result[task]["probabilities"]
        assert isinstance(probs, dict)
        assert len(probs) > 0
        total_p = sum(probs.values())
        assert abs(total_p - 1.0) < 0.05
        assert result[task]["prediction"] in probs

