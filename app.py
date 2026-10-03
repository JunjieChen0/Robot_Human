"""Robot_Human Streamlit 交互原型。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CLASS_NAMES,
    CLASS_NAMES_ZH,
    FEATURE_NAMES,
    MODELS_DIR,
    ROOT_DIR,
    SAMPLE_DIR,
)
from src.validation import validate_feature_frame


st.set_page_config(
    page_title="手机活动识别",
    page_icon="📱",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "final_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError("尚未找到模型，请先运行 python -m src.train。")
    return joblib.load(model_path)


@st.cache_data
def load_json(path_string: str) -> dict:
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


@st.cache_data
def load_demo_samples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "demo_samples.csv")


@st.cache_data
def load_template() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "input_template.csv")


@st.cache_data
def load_feature_metadata() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "feature_metadata.csv")


def model_classes(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and hasattr(classifier, "classes_"):
        return np.asarray(classifier.classes_)
    raise AttributeError("模型没有找到类别信息。")


def predict_one(model, values: pd.DataFrame) -> dict[str, object]:
    classes = model_classes(model)
    prediction = int(model.predict(values)[0])
    probabilities = np.asarray(model.predict_proba(values)[0], dtype=float)
    probability_by_label = {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }
    ranked = sorted(
        probability_by_label.items(), key=lambda item: item[1], reverse=True
    )
    return {
        "prediction": prediction,
        "probability_by_label": probability_by_label,
        "top_probability": ranked[0][1],
        "second_probability": ranked[1][1] if len(ranked) > 1 else 0.0,
    }


def probability_table(result: dict[str, object]) -> pd.DataFrame:
    probability_by_label = result["probability_by_label"]
    rows = []
    for label_id in CLASS_NAMES:
        rows.append(
            {
                "活动": CLASS_NAMES_ZH[label_id],
                "英文标签": CLASS_NAMES[label_id],
                "模型估计概率": float(probability_by_label.get(label_id, 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("模型估计概率", ascending=False)


def show_feature_explanation(metadata: dict, feature_metadata: pd.DataFrame) -> None:
    st.subheader("解释反馈")
    st.write(
        "模型使用窗口内身体加速度和陀螺仪的均值、变化程度和整体强度进行判断。"
        "下表是模型整体的特征重要性，不等于某一次预测的因果解释。"
    )
    importance = pd.DataFrame(metadata.get("global_feature_importance", []))
    if importance.empty:
        st.info("当前模型没有保存可展示的特征重要性。")
        return
    explanation = importance.head(8).merge(
        feature_metadata,
        left_on="feature",
        right_on="feature",
        how="left",
    )
    explanation["importance"] = explanation["importance"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        explanation[["feature", "signal", "metric", "description", "unit", "importance"]],
        hide_index=True,
        width="stretch",
    )


def show_result(
    result: dict[str, object],
    warnings: list[str],
    metadata: dict,
    feature_metadata: pd.DataFrame,
) -> None:
    prediction = int(result["prediction"])
    top_probability = float(result["top_probability"])
    second_probability = float(result["second_probability"])
    st.subheader("预测结果")
    result_col, probability_col, gap_col = st.columns(3)
    result_col.metric("预测活动", CLASS_NAMES_ZH[prediction])
    probability_col.metric("最高类别概率", f"{top_probability:.1%}")
    gap_col.metric("最高与第二高差值", f"{top_probability - second_probability:.1%}")
    st.caption(
        f"英文标签：{CLASS_NAMES[prediction]}。该结果仅表示模型对当前输入的类别估计。"
    )

    table = probability_table(result)
    st.dataframe(
        table.style.format({"模型估计概率": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )
    st.bar_chart(table.set_index("活动")["模型估计概率"], y_label="概率")

    if warnings:
        st.warning("当前输入存在训练范围外特征，预测结果需要谨慎理解。")
        for warning in warnings:
            st.caption(f"提示：{warning}")
    else:
        st.success("当前输入的所有特征都处于训练数据的最小/最大范围内。")

    show_feature_explanation(metadata, feature_metadata)


def show_comparison(previous: dict[str, object], current: dict[str, object]) -> None:
    previous_prediction = int(previous["prediction"])
    current_prediction = int(current["prediction"])
    comparison = pd.DataFrame(
        [
            {
                "项目": "预测活动",
                "修改前": CLASS_NAMES_ZH[previous_prediction],
                "修改后": CLASS_NAMES_ZH[current_prediction],
            },
            {
                "项目": "最高类别概率",
                "修改前": f"{float(previous['top_probability']):.1%}",
                "修改后": f"{float(current['top_probability']):.1%}",
            },
        ]
    )
    st.subheader("修改前后对比")
    st.dataframe(comparison, hide_index=True, width="stretch")


def read_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:  # Streamlit 页面需要将格式错误转成用户可理解的提示
        st.error(f"无法读取 CSV 文件：{exc}")
        return None


def main() -> None:
    st.title("📱 手机传感器日常活动识别")
    st.write(
        "选择一个传感器窗口样例或上传 18 个统计特征，查看模型对六类日常活动的估计结果。"
    )
    st.info(
        "课程实验声明：预测概率不是绝对正确率，结果不代表医学诊断、疾病筛查或健康风险结论。"
    )

    required_paths = [
        MODELS_DIR / "final_model.joblib",
        MODELS_DIR / "model_metadata.json",
        MODELS_DIR / "feature_ranges.json",
        SAMPLE_DIR / "demo_samples.csv",
        SAMPLE_DIR / "input_template.csv",
    ]
    missing_paths = [str(path.relative_to(ROOT_DIR)) for path in required_paths if not path.is_file()]
    if missing_paths:
        st.error("原型所需文件尚未生成：" + ", ".join(missing_paths))
        st.code("python -m src.preprocess\npython -m src.train\npython -m src.evaluate")
        st.stop()

    try:
        model = load_model()
        metadata = load_json(str(MODELS_DIR / "model_metadata.json"))
        feature_ranges = load_json(str(MODELS_DIR / "feature_ranges.json"))
        feature_metadata = load_feature_metadata()
    except Exception as exc:
        st.error(f"加载模型或元数据失败：{exc}")
        st.stop()

    st.sidebar.header("输入方式")
    input_mode = st.sidebar.radio(
        "选择数据输入方式",
        ["选择演示样例", "上传特征 CSV"],
    )

    if input_mode == "选择演示样例":
        demo_samples = load_demo_samples()
        sample_labels = demo_samples["sample_id"].tolist()
        selected_sample = st.sidebar.selectbox("选择样例", sample_labels)
        selected_row = demo_samples.loc[
            demo_samples["sample_id"].eq(selected_sample), FEATURE_NAMES
        ]
        base_values = selected_row.reset_index(drop=True)
        st.sidebar.caption("样例来自测试数据的单个窗口，仅用于课程演示。")
    else:
        uploaded_file = st.sidebar.file_uploader("上传一行或多行特征 CSV", type=["csv"])
        if uploaded_file is None:
            st.sidebar.caption("尚未上传文件，先显示模板中位数；可直接修改后预测。")
            base_values = load_template()[FEATURE_NAMES]
        else:
            uploaded_frame = read_uploaded_file(uploaded_file)
            if uploaded_frame is None:
                st.stop()
            validated_all, upload_errors, upload_warnings = validate_feature_frame(
                uploaded_frame, feature_ranges
            )
            if upload_errors:
                for error in upload_errors:
                    st.error(error)
                st.stop()
            for warning in upload_warnings:
                st.warning(warning)
            if len(validated_all) > 1:
                selected_index = st.sidebar.selectbox(
                    "选择要预测的行",
                    list(range(len(validated_all))),
                    format_func=lambda value: f"第 {value + 1} 行",
                )
            else:
                selected_index = 0
            base_values = validated_all.iloc[[selected_index]].reset_index(drop=True)

    st.subheader("输入数据")
    st.caption(
        "可以直接修改下表中的任意数值。均值/标准差/RMS 分别表示窗口平均水平、变化程度和整体强度。"
    )
    column_config = {}
    metadata_by_feature = feature_metadata.set_index("feature").to_dict("index")
    for feature in FEATURE_NAMES:
        info = metadata_by_feature.get(feature, {})
        column_config[feature] = st.column_config.NumberColumn(
            label=feature,
            help=(
                f"{info.get('description', '')}；单位：{info.get('unit', '无')}"
            ),
            format="%.6f",
        )

    with st.form("prediction_form"):
        edited_values = st.data_editor(
            base_values,
            hide_index=True,
            num_rows="fixed",
            width="stretch",
            column_config=column_config,
        )
        submitted = st.form_submit_button("预测 / 重新预测", type="primary")

    st.download_button(
        "下载输入模板",
        data=load_template().to_csv(index=False).encode("utf-8-sig"),
        file_name="input_template.csv",
        mime="text/csv",
    )

    if not submitted:
        st.subheader("使用说明")
        st.markdown(
            "1. 选择样例或上传 CSV。\n"
            "2. 检查并修改输入特征。\n"
            "3. 点击“预测 / 重新预测”。\n"
            "4. 查看类别概率、解释和训练范围提示。"
        )
        return

    values, errors, warnings = validate_feature_frame(edited_values, feature_ranges)
    if errors:
        for error in errors:
            st.error(error)
        return
    for warning in warnings:
        st.warning(warning)

    previous_result = st.session_state.get("last_prediction")
    result = predict_one(model, values)
    st.session_state["last_prediction"] = result
    show_result(result, warnings, metadata, feature_metadata)
    if previous_result is not None:
        show_comparison(previous_result, result)

    with st.expander("模型与数据适用范围", expanded=False):
        st.write(
            "本模型在 UCI HAR 数据集的特定受试者、腰部佩戴手机、固定采样条件和六类活动上训练。"
            "换用其他设备、佩戴位置、人群或未覆盖活动时，预测可靠性需要另行验证。"
        )
        st.write(
            f"当前原型使用模型：{metadata.get('model_name', '未知')}；"
            "输入特征来自 2.56 秒窗口的统计特征。"
        )


if __name__ == "__main__":
    main()
