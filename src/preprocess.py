"""UCI HAR 数据读取、质量检查、特征提取和受试者分组划分。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CLASS_NAMES,
    CLASS_NAMES_ZH,
    FEATURE_NAMES,
    FEATURE_UNITS,
    METRIC_DESCRIPTIONS,
    METRICS,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORT_TABLES_DIR,
    SAMPLE_DIR,
    SENSOR_CHANNELS,
    UCI_DIR,
)


def _read_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _load_raw_split(split: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if split not in {"train", "test"}:
        raise ValueError("split 必须是 train 或 test。")

    signal_dir = UCI_DIR / split / "Inertial Signals"
    signals = {
        channel: _read_matrix(signal_dir / f"{channel}_{split}.txt")
        for channel in SENSOR_CHANNELS
    }
    labels = np.loadtxt(UCI_DIR / split / f"y_{split}.txt", dtype=np.int64)
    subjects = np.loadtxt(UCI_DIR / split / f"subject_{split}.txt", dtype=np.int64)
    labels = np.atleast_1d(labels)
    subjects = np.atleast_1d(subjects)

    row_counts = {channel: values.shape[0] for channel, values in signals.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"{split} 集六个传感器文件行数不一致：{row_counts}")
    n_rows = next(iter(row_counts.values()))
    if labels.shape[0] != n_rows or subjects.shape[0] != n_rows:
        raise ValueError(
            f"{split} 集信号、标签和受试者数量不一致："
            f"signals={n_rows}, labels={labels.shape[0]}, subjects={subjects.shape[0]}"
        )
    if any(values.shape[1] != 128 for values in signals.values()):
        raise ValueError(f"{split} 集每个窗口应有 128 个时间点。")

    return signals, labels, subjects


def extract_features(signals: dict[str, np.ndarray]) -> pd.DataFrame:
    """从每个通道的 128 点窗口提取 mean、std 和 RMS。"""

    features: dict[str, np.ndarray] = {}
    for channel in SENSOR_CHANNELS:
        values = np.asarray(signals[channel], dtype=np.float64)
        features[f"{channel}_mean"] = np.mean(values, axis=1)
        features[f"{channel}_std"] = np.std(values, axis=1, ddof=0)
        features[f"{channel}_rms"] = np.sqrt(np.mean(values**2, axis=1))
    return pd.DataFrame(features, columns=FEATURE_NAMES)


def _build_split_frame(split: str) -> pd.DataFrame:
    signals, labels, subjects = _load_raw_split(split)
    frame = extract_features(signals)
    frame.insert(0, "label_id", labels)
    frame.insert(0, "activity", [CLASS_NAMES.get(int(label), "UNKNOWN") for label in labels])
    frame.insert(0, "subject_id", subjects)
    frame.insert(0, "source_split", split)
    return frame


def assign_group_splits(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """将官方训练集按 subject_id 分成 train/validation。

    官方测试集保留为 test；该函数主要用于测试和复用，因此要求输入中
    已有 source_split 列。
    """

    result = frame.copy()
    result["split"] = "test"
    train_mask = result["source_split"].eq("train").to_numpy()
    train_frame = result.loc[train_mask]
    if train_frame.empty:
        raise ValueError("没有找到官方训练数据。")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_state,
    )
    train_indices, validation_indices = next(
        splitter.split(
            train_frame[FEATURE_NAMES],
            train_frame["label_id"],
            groups=train_frame["subject_id"],
        )
    )
    train_positions = train_frame.index.to_numpy()
    result.loc[train_positions[train_indices], "split"] = "train"
    result.loc[train_positions[validation_indices], "split"] = "validation"

    group_sets = {
        name: set(result.loc[result["split"].eq(name), "subject_id"].unique())
        for name in ("train", "validation", "test")
    }
    if group_sets["train"] & group_sets["validation"]:
        raise AssertionError("训练集和验证集存在受试者交集。")
    if group_sets["train"] & group_sets["test"]:
        raise AssertionError("训练集和测试集存在受试者交集。")
    if group_sets["validation"] & group_sets["test"]:
        raise AssertionError("验证集和测试集存在受试者交集。")

    return result


def _feature_ranges(frame: pd.DataFrame, split_name: str = "train") -> dict[str, dict[str, float]]:
    train_values = frame.loc[frame["split"].eq(split_name), FEATURE_NAMES]
    if train_values.empty:
        raise ValueError(f"没有找到 {split_name} 数据，无法计算特征范围。")
    result: dict[str, dict[str, float]] = {}
    for column in FEATURE_NAMES:
        values = train_values[column].astype(float)
        result[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(values.quantile(0.01)),
            "p99": float(values.quantile(0.99)),
            "median": float(values.median()),
        }
    return result


def _quality_report(frame: pd.DataFrame) -> dict[str, object]:
    feature_values = frame[FEATURE_NAMES]
    missing_by_column = feature_values.isna().sum()
    non_finite_by_column = pd.Series(
        {
            column: int((~np.isfinite(feature_values[column].to_numpy(dtype=float))).sum())
            for column in FEATURE_NAMES
        }
    )

    class_counts = (
        frame["label_id"].value_counts().sort_index().to_dict()
    )
    split_counts = frame["split"].value_counts().to_dict()
    subject_counts = frame.groupby("split")["subject_id"].nunique().to_dict()
    train_values = frame.loc[frame["split"].eq("train"), FEATURE_NAMES]
    iqr_outlier_counts: dict[str, int] = {}
    for column in FEATURE_NAMES:
        q1 = float(train_values[column].quantile(0.25))
        q3 = float(train_values[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outside = (feature_values[column] < lower) | (feature_values[column] > upper)
        iqr_outlier_counts[column] = int(outside.sum())
    return {
        "rows": int(len(frame)),
        "feature_count": len(FEATURE_NAMES),
        "missing_total": int(missing_by_column.sum()),
        "missing_by_column": {key: int(value) for key, value in missing_by_column.items()},
        "non_finite_total": int(non_finite_by_column.sum()),
        "non_finite_by_column": {
            key: int(value) for key, value in non_finite_by_column.items()
        },
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "duplicate_feature_rows": int(frame.duplicated(subset=FEATURE_NAMES).sum()),
        "iqr_outlier_rule": "以训练集 Q1-1.5*IQR 和 Q3+1.5*IQR 作为统计标记，不自动删除",
        "iqr_outlier_counts": iqr_outlier_counts,
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "subject_counts": {key: int(value) for key, value in subject_counts.items()},
        "subject_overlap_checked": True,
    }


def _save_feature_metadata() -> None:
    rows = []
    for channel in SENSOR_CHANNELS:
        unit = FEATURE_UNITS[channel]
        for metric in METRICS:
            rows.append(
                {
                    "feature": f"{channel}_{metric}",
                    "signal": channel,
                    "metric": metric,
                    "description": METRIC_DESCRIPTIONS[metric],
                    "unit": unit,
                }
            )
    pd.DataFrame(rows).to_csv(SAMPLE_DIR / "feature_metadata.csv", index=False)


def build_processed_dataset() -> pd.DataFrame:
    """读取、处理、划分并保存完整数据集。"""

    for directory in (PROCESSED_DIR, SAMPLE_DIR, REPORT_TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    train_frame = _build_split_frame("train")
    test_frame = _build_split_frame("test")
    frame = pd.concat([train_frame, test_frame], ignore_index=True)
    frame = assign_group_splits(frame)

    if not set(frame["label_id"].unique()).issubset(set(CLASS_NAMES)):
        raise ValueError("数据中存在未知活动标签。")

    frame.to_csv(PROCESSED_DIR / "features.csv", index=False)
    frame.groupby(["split", "subject_id"], as_index=False).size().rename(
        columns={"size": "sample_count"}
    ).to_csv(PROCESSED_DIR / "split_metadata.csv", index=False)

    quality = _quality_report(frame)
    (PROCESSED_DIR / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    feature_ranges = _feature_ranges(frame)
    (PROCESSED_DIR / "feature_ranges.json").write_text(
        json.dumps(feature_ranges, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    feature_stats = frame[FEATURE_NAMES].describe().T.reset_index(names="feature")
    feature_stats.to_csv(REPORT_TABLES_DIR / "feature_statistics.csv", index=False)

    demo_rows = []
    for label_id, activity in CLASS_NAMES.items():
        candidates = frame[
            frame["split"].eq("test") & frame["label_id"].eq(label_id)
        ]
        if candidates.empty:
            raise ValueError(f"测试集中没有活动类别 {activity}，无法生成演示样例。")
        selected = candidates.iloc[0]
        demo_rows.append(
            {
                "sample_id": f"demo_{label_id}_{activity.lower()}",
                **{feature: float(selected[feature]) for feature in FEATURE_NAMES},
                "reference_label": activity,
            }
        )
    pd.DataFrame(demo_rows).to_csv(SAMPLE_DIR / "demo_samples.csv", index=False)

    template_values = frame.loc[frame["split"].eq("train"), FEATURE_NAMES].median()
    pd.DataFrame([{feature: float(template_values[feature]) for feature in FEATURE_NAMES}]).to_csv(
        SAMPLE_DIR / "input_template.csv", index=False
    )
    _save_feature_metadata()

    print(f"处理完成：{PROCESSED_DIR / 'features.csv'}")
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return frame


def main() -> None:
    build_processed_dataset()


if __name__ == "__main__":
    main()
