import pandas as pd

from src.config import FEATURE_NAMES
from src.validation import validate_feature_frame


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame([{feature: 0.0 for feature in FEATURE_NAMES}])


def test_validation_accepts_numeric_features():
    values, errors, warnings = validate_feature_frame(valid_frame())
    assert not errors
    assert not warnings
    assert list(values.columns) == FEATURE_NAMES


def test_validation_reports_missing_and_non_numeric_values():
    frame = valid_frame().drop(columns=[FEATURE_NAMES[0]])
    values, errors, warnings = validate_feature_frame(frame)
    assert values.empty
    assert any("缺少必要特征列" in error for error in errors)


def test_validation_reports_non_numeric_values():
    frame = valid_frame().astype(object)
    frame.loc[0, FEATURE_NAMES[0]] = "not-a-number"
    values, errors, warnings = validate_feature_frame(frame)
    assert any("非数值" in error for error in errors)


def test_validation_warns_when_value_is_outside_training_range():
    frame = valid_frame()
    ranges = {
        feature: {"min": -1.0, "max": 1.0}
        for feature in FEATURE_NAMES
    }
    frame.loc[0, FEATURE_NAMES[0]] = 2.0
    values, errors, warnings = validate_feature_frame(frame, ranges)
    assert not errors
    assert values.loc[0, FEATURE_NAMES[0]] == 2.0
    assert any(FEATURE_NAMES[0] in warning for warning in warnings)
