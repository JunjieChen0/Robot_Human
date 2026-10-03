"""交互输入和数据格式验证函数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES


ALLOWED_METADATA_COLUMNS = {"sample_id", "reference_label"}


def validate_feature_frame(
    frame: pd.DataFrame,
    feature_ranges: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """验证一个包含模型输入特征的 DataFrame。

    返回值为：清洗后的特征 DataFrame、错误信息列表、警告信息列表。
    训练范围之外的数值只产生警告，不直接阻止预测，因为极端值可能是
    合法但未被训练数据充分覆盖的传感器输入。
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=FEATURE_NAMES), ["没有可供预测的数据行。"], warnings

    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        errors.append("缺少必要特征列：" + ", ".join(missing))
        return pd.DataFrame(columns=FEATURE_NAMES), errors, warnings

    unexpected = [
        name
        for name in frame.columns
        if name not in FEATURE_NAMES and name not in ALLOWED_METADATA_COLUMNS
    ]
    if unexpected:
        warnings.append("检测到额外列，将不会送入模型：" + ", ".join(unexpected))

    values = frame[FEATURE_NAMES].copy()
    for column in FEATURE_NAMES:
        original = values[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_mask = converted.isna() & original.notna()
        if invalid_mask.any():
            errors.append(f"特征列 {column} 含有非数值内容。")
        values[column] = converted

    null_columns = [column for column in FEATURE_NAMES if values[column].isna().any()]
    if null_columns:
        errors.append("以下特征含有空值：" + ", ".join(null_columns))

    if not values.empty:
        numeric_values = values.to_numpy(dtype=float)
        finite_mask = np.isfinite(numeric_values)
        if not finite_mask.all():
            errors.append("输入中含有 NaN、正无穷或负无穷，无法进行预测。")

    if not errors and feature_ranges:
        for column in FEATURE_NAMES:
            range_info: dict[str, Any] = feature_ranges.get(column, {})
            lower = range_info.get("min")
            upper = range_info.get("max")
            if lower is None or upper is None:
                continue
            outside = (values[column] < float(lower)) | (values[column] > float(upper))
            if outside.any():
                warnings.append(
                    f"{column} 有 {int(outside.sum())} 个值超出训练数据的最小/最大范围。"
                )

    return values, errors, warnings
