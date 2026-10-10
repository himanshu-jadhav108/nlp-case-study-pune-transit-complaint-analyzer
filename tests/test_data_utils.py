import pandas as pd

from src.data_utils import REQUIRED_COLUMNS, TEXT_COLUMN, build_synthetic_dataset, prepare_batch_frame, validate_dataset


def test_empty_complaint_validation():
    data = build_synthetic_dataset().iloc[:2].copy()
    data.loc[data.index[0], TEXT_COLUMN] = "   "
    valid, notes, counts = validate_dataset(data)
    assert len(valid) == 1
    assert counts["empty_text_rows"] == 1
    assert any("empty" in n for n in notes)


def test_csv_schema_validation():
    valid, errors, counts = validate_dataset(pd.DataFrame({"wrong": ["x"]}))
    assert valid.empty
    assert "complaint_text" in errors[0]


def test_duplicates_and_unknown_labels_are_reported_and_removed():
    data = build_synthetic_dataset().iloc[:2].copy()
    data = pd.concat([data, data.iloc[[0]], data.iloc[[1]].assign(category="Mystery")], ignore_index=True)
    valid, notes, counts = validate_dataset(data)
    assert len(valid) == 2
    assert counts["duplicate_rows"] == 1
    assert counts["unknown_label_rows"] == 1


def test_synthetic_data_is_balanced_and_reproducible():
    first, second = build_synthetic_dataset(), build_synthetic_dataset()
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == REQUIRED_COLUMNS
    assert first.category.value_counts().nunique() == 1


def test_batch_upload_valid_and_invalid_rows():
    clean, skipped = prepare_batch_frame(pd.DataFrame({TEXT_COLUMN: [" late bus ", "", None], "id": [1, 2, 3]}))
    assert skipped == 2
    assert clean[TEXT_COLUMN].tolist() == ["late bus"]
    assert clean["id"].tolist() == [1]
