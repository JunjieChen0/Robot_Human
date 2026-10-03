import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES, SENSOR_CHANNELS
from src.preprocess import assign_group_splits, extract_features


def test_extract_features_has_expected_columns_and_values():
    signals = {
        channel: np.arange(256, dtype=float).reshape(2, 128) + index
        for index, channel in enumerate(SENSOR_CHANNELS)
    }
    result = extract_features(signals)

    assert list(result.columns) == FEATURE_NAMES
    assert result.shape == (2, 18)
    assert result.loc[0, "body_acc_x_mean"] == np.mean(signals["body_acc_x"][0])
    assert result.loc[0, "body_acc_x_rms"] == np.sqrt(
        np.mean(signals["body_acc_x"][0] ** 2)
    )


def test_assign_group_splits_does_not_mix_subjects():
    rows = []
    for subject_id in range(1, 6):
        for sample_id in range(2):
            rows.append(
                {
                    "source_split": "train" if subject_id < 5 else "test",
                    "subject_id": subject_id,
                    "label_id": 1 + sample_id,
                    **{feature: float(subject_id + sample_id) for feature in FEATURE_NAMES},
                }
            )
    result = assign_group_splits(pd.DataFrame(rows), random_state=42)
    groups = {
        split: set(result.loc[result["split"].eq(split), "subject_id"])
        for split in ("train", "validation", "test")
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert groups["test"] == {5}
