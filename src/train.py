"""训练并比较逻辑回归和随机森林模型。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASS_NAMES,
    CLASS_NAMES_ZH,
    FEATURE_NAMES,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
)


matplotlib.use("Agg")


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """返回两种待比较的模型。"""

    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def _metric_row(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    training_samples: int,
    validation_samples: int,
) -> dict[str, float | int | str]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES),
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(CLASS_NAMES),
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "training_samples": int(training_samples),
        "validation_samples": int(validation_samples),
    }


def _training_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for feature in FEATURE_NAMES:
        values = frame[feature].astype(float)
        ranges[feature] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(values.quantile(0.01)),
            "p99": float(values.quantile(0.99)),
            "median": float(values.median()),
        }
    return ranges


def _feature_importance(model: Pipeline) -> list[dict[str, float | str]]:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.mean(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
    else:
        values = np.zeros(len(FEATURE_NAMES), dtype=float)

    total = float(values.sum())
    if total > 0:
        values = values / total
    ranking = sorted(
        (
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(FEATURE_NAMES, values)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranking


def _save_importance_plot(importance: list[dict[str, float | str]], model_name: str) -> None:
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_items = importance[:10][::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(
        [str(item["feature"]) for item in top_items],
        [float(item["importance"]) for item in top_items],
        color="#4c78a8",
    )
    plt.xlabel("Normalized global importance")
    plt.title(f"Top features - {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(REPORT_FIGURES_DIR / f"feature_importance_{safe_name}.png", dpi=160)
    plt.close()


def train_and_compare() -> dict[str, object]:
    """训练验证模型，选择最终模型并保存模型文件。"""

    for directory in (MODELS_DIR, REPORT_TABLES_DIR, REPORT_FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data_path = PROCESSED_DIR / "features.csv"
    if not data_path.is_file():
        raise FileNotFoundError("没有找到处理后的数据，请先运行 python -m src.preprocess。")

    frame = pd.read_csv(data_path)
    train_frame = frame[frame["split"].eq("train")]
    validation_frame = frame[frame["split"].eq("validation")]
    fit_frame = frame[frame["split"].isin(["train", "validation"])]

    X_train = train_frame[FEATURE_NAMES]
    y_train = train_frame["label_id"].astype(int)
    X_validation = validation_frame[FEATURE_NAMES]
    y_validation = validation_frame["label_id"].astype(int)

    validation_rows: list[dict[str, float | int | str]] = []
    validation_reports: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        validation_rows.append(
            _metric_row(
                model_name,
                y_validation,
                predictions,
                len(train_frame),
                len(validation_frame),
            )
        )
        validation_reports[model_name] = classification_report(
            y_validation,
            predictions,
            labels=list(CLASS_NAMES),
            target_names=[CLASS_NAMES[label] for label in CLASS_NAMES],
            output_dict=True,
            zero_division=0,
        )

    comparison = pd.DataFrame(validation_rows).sort_values(
        by=["f1_macro", "accuracy"], ascending=False
    )
    comparison.to_csv(REPORT_TABLES_DIR / "model_comparison.csv", index=False)
    (REPORT_TABLES_DIR / "validation_classification_reports.json").write_text(
        json.dumps(validation_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_model_name = str(comparison.iloc[0]["model"])
    final_model = build_models()[selected_model_name]
    final_model.fit(fit_frame[FEATURE_NAMES], fit_frame["label_id"].astype(int))
    joblib.dump(final_model, MODELS_DIR / "final_model.joblib")

    importance = _feature_importance(final_model)
    _save_importance_plot(importance, selected_model_name)
    ranges = _training_ranges(fit_frame)
    (MODELS_DIR / "feature_ranges.json").write_text(
        json.dumps(ranges, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "model_name": selected_model_name,
        "feature_names": FEATURE_NAMES,
        "label_mapping": {
            str(label): {
                "name": CLASS_NAMES[label],
                "name_zh": CLASS_NAMES_ZH[label],
            }
            for label in CLASS_NAMES
        },
        "random_state": RANDOM_STATE,
        "validation_selection_metric": "f1_macro",
        "training_samples_before_final_fit": int(len(train_frame)),
        "validation_samples": int(len(validation_frame)),
        "final_fit_samples": int(len(fit_frame)),
        "global_feature_importance": importance,
        "model_parameters": {
            key: str(value) for key, value in final_model.get_params().items()
        },
        "probability_note": "predict_proba 输出为模型类别概率估计，未自动解释为校准后的真实置信度。",
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("验证集模型比较：")
    print(comparison.to_string(index=False))
    print(f"最终模型：{selected_model_name}")
    return metadata


def main() -> None:
    train_and_compare()


if __name__ == "__main__":
    main()
