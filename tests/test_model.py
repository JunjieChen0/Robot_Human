import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES
from src.train import build_models


def test_both_models_fit_and_return_six_class_probabilities():
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for label in range(1, 7):
        for _ in range(5):
            rows.append(rng.normal(loc=label, scale=0.1, size=len(FEATURE_NAMES)))
            labels.append(label)
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels)

    for model in build_models().values():
        model.fit(X, y)
        probabilities = model.predict_proba(X.iloc[:2])
        assert probabilities.shape == (2, 6)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
