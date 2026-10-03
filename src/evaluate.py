"""在独立测试集上评价最终模型并生成图表。"""

from __future__ import annotations

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    CLASS_NAMES,
    FEATURE_NAMES,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
)


matplotlib.use("Agg")


def _save_confusion_matrix(matrix: np.ndarray, normalized: bool = False) -> None:
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [CLASS_NAMES[label] for label in CLASS_NAMES]
    plt.figure(figsize=(9, 7))
    image = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            color = "white" if value > threshold else "black"
            plt.text(column, row, text, ha="center", va="center", color=color)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    title = "Normalized confusion matrix" if normalized else "Confusion matrix"
    plt.title(title)
    plt.tight_layout()
    filename = "normalized_confusion_matrix.png" if normalized else "confusion_matrix.png"
    plt.savefig(REPORT_FIGURES_DIR / filename, dpi=160)
    plt.close()


def _save_class_distribution(frame: pd.DataFrame) -> None:
    counts = (
        frame["activity"]
        .value_counts()
        .reindex([CLASS_NAMES[label] for label in CLASS_NAMES])
        .fillna(0)
    )
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values, color="#59a14f")
    plt.ylabel("Samples")
    plt.xlabel("Activity")
    plt.title("Class distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()


def evaluate_final_model() -> dict[str, object]:
    for directory in (REPORT_TABLES_DIR, REPORT_FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data_path = PROCESSED_DIR / "features.csv"
    model_path = MODELS_DIR / "final_model.joblib"
    if not data_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("请先运行预处理和训练步骤。")

    frame = pd.read_csv(data_path)
    test_frame = frame[frame["split"].eq("test")].copy()
    model = joblib.load(model_path)
    y_true = test_frame["label_id"].astype(int).to_numpy()
    predictions = model.predict(test_frame[FEATURE_NAMES])
    probabilities = model.predict_proba(test_frame[FEATURE_NAMES])

    labels = list(CLASS_NAMES)
    target_names = [CLASS_NAMES[label] for label in labels]
    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T.reset_index(names="class")
    report_frame.to_csv(REPORT_TABLES_DIR / "test_classification_report.csv", index=False)

    matrix = confusion_matrix(y_true, predictions, labels=labels)
    normalized_matrix = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
    _save_confusion_matrix(matrix, normalized=False)
    _save_confusion_matrix(normalized_matrix, normalized=True)
    _save_class_distribution(test_frame)

    class_columns = [f"prob_{CLASS_NAMES[label]}" for label in labels]
    prediction_frame = test_frame[["subject_id", "activity", "label_id"]].copy()
    prediction_frame["predicted_label_id"] = predictions
    prediction_frame["predicted_activity"] = [CLASS_NAMES[int(value)] for value in predictions]
    prediction_frame["top_probability"] = probabilities.max(axis=1)
    prediction_frame = pd.concat(
        [prediction_frame.reset_index(names="sample_index"), pd.DataFrame(probabilities, columns=class_columns)],
        axis=1,
    )
    prediction_frame.to_csv(REPORT_TABLES_DIR / "test_predictions.csv", index=False)

    error_mask = y_true != predictions
    error_frame = prediction_frame.loc[error_mask].copy()
    error_frame["true_activity"] = error_frame["activity"]
    error_frame["predicted_activity"] = error_frame["predicted_activity"]
    probability_values = probabilities[error_mask]
    if len(error_frame):
        order = np.argsort(-probability_values, axis=1)
        error_frame["second_probability"] = [
            float(probability_values[row, order[row, 1]]) for row in range(len(order))
        ]
        error_frame["second_activity"] = [
            CLASS_NAMES[int(model.classes_[order[row, 1]])] for row in range(len(order))
        ]
    else:
        error_frame["second_probability"] = pd.Series(dtype=float)
        error_frame["second_activity"] = pd.Series(dtype=str)
    feature_error_frame = test_frame.loc[error_mask, FEATURE_NAMES].reset_index(drop=True)
    error_frame = pd.concat([error_frame.reset_index(drop=True), feature_error_frame], axis=1)
    error_frame.to_csv(REPORT_TABLES_DIR / "error_cases.csv", index=False)

    summary = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "test_samples": int(len(test_frame)),
        "error_samples": int(error_mask.sum()),
        "error_rate": float(error_mask.mean()),
        "confusion_matrix": matrix.tolist(),
    }
    (REPORT_TABLES_DIR / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(report_frame.to_string(index=False))
    return summary


def main() -> None:
    evaluate_final_model()


if __name__ == "__main__":
    main()
