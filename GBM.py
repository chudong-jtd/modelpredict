# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import tempfile
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from shap import TreeExplainer
from streamlit.components.v1 import html as st_html

# 类型定义
SklearnModel = RandomForestClassifier
SklearnScaler = StandardScaler


def load_model_and_scaler() -> Tuple[SklearnModel, SklearnScaler]:
    """加载模型和标准化器"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(current_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        st.stop()


def validate_inputs(PH: float, PO2: float, Lac: float):
    """临床参数验证"""
    alerts = []
    if PH < 7.2:
        alerts.append("严重酸中毒（pH < 7.2）")
    elif PH < 7.35:
        alerts.append("轻度酸中毒（7.2 ≤ pH < 7.35）")
    elif PH > 7.45:
        alerts.append("碱中毒（pH > 7.45）")

    if PO2 < 50:
        alerts.append("严重低氧（PO₂ < 50 mmHg）")
    elif PO2 < 60:
        alerts.append("轻度低氧（50 ≤ PO₂ < 60 mmHg）")
    elif PO2 > 300:
        alerts.append("异常高氧（PO₂ > 300 mmHg）")

    if Lac > 4:
        alerts.append("严重高乳酸（Lac > 4 mmol/L）")
    elif Lac > 2:
        alerts.append("高乳酸（2 < Lac ≤ 4 mmol/L）")

    for alert in alerts:
        if "严重" in alert:
            st.error(f"🚨 {alert}")
        elif "异常" in alert:
            st.error(f"⚠️ {alert}")
        else:
            st.warning(f"⚠️ {alert}")


def make_prediction(model: SklearnModel, scaler: SklearnScaler, input_data: pd.DataFrame) -> float:
    """执行预测"""
    try:
        scaled_data = scaler.transform(input_data)
        proba = model.predict_proba(scaled_data)
        return proba[0][1]
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        st.stop()


def generate_shap_plot(model: SklearnModel, scaler: SklearnScaler, input_data: pd.DataFrame) -> str:
    """生成优化后的SHAP可视化"""
    try:
        scaled_data = scaler.transform(input_data)
        explainer = TreeExplainer(model)
        shap_values = explainer(scaled_data)

        fig = shap.plots.force(
            base_value=explainer.expected_value[1],
            shap_values=shap_values.values[..., 1],
            features=scaled_data[0],
            feature_names=input_data.columns.tolist(),
            matplotlib=False,
            plot_cmap="coolwarm",
            text_rotation=15,
            figsize=(12, 6)
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            shap.save_html(tmp.name, fig)

            with open(tmp.name, "r", encoding="utf-8") as f:
                html_content = f.read()

            custom_style = """
            <style>
                #container { width: 100% !important; height: 550px !important; padding: 15px !important; }
                .feature-name { font-size: 11px !important; transform: translateY(4px) rotate(15deg) !important; opacity: 0.9 !important; }
                .value { font-size: 10px !important; transform: translateY(-2px) !important; opacity: 0.8 !important; }
                .base-value, .output-value { font-size: 12px !important; font-weight: 600 !important; transform: translate(5px, 15px) !important; }
                .arrow { stroke-width: 1.2 !important; opacity: 0.7 !important; }
                .color-scale { transform: translateY(10px) !important; }
                .hover-info, .x-axis-label { display: none !important; }
                .force-plot .labels > * { margin: 2px 0 !important; }
            </style>
            """
            html_content = html_content.replace('</head>', f'{custom_style}</head>')
            return html_content
    finally:
        if tmp: os.remove(tmp.name)


def main():
    model, scaler = load_model_and_scaler()

    st.set_page_config(
        page_title="临床预测系统",
        layout="wide",
        page_icon="🏥"
    )
    st.title("🏥 危重症AKI死亡风险预测系统")

    # 完整的侧边栏输入组件
    with st.sidebar:
        st.header("患者生理参数")
        inputs = {
            'SBP': st.slider("收缩压 (mmHg)", 30.0, 250.0, 120.0, 1.0),
            'UO': st.slider("尿量 (mL/24h)", 0.0, 10000.0, 1500.0, 1.0),
            'PLT': st.slider("血小板计数 (×10⁹/L)", 0.0, 800.0, 200.0, 1.0),
            'Na': st.slider("血钠 (mmol/L)", 100.0, 190.0, 140.0, 0.1),
            'LDH': st.slider("乳酸脱氢酶 (U/L)", 50.0, 32000.0, 200.0, 1.0),
            'PH': st.slider("动脉pH值", 6.7, 7.6, 7.4, 0.01),
            'PO2': st.slider("血氧分压 (mmHg)", 10.0, 600.0, 95.0, 0.1),
            'Lac': st.slider("动脉乳酸 (mmol/L)", 0.5, 30.0, 1.2, 0.1),
            'BE': st.slider("碱剩余 (mmol/L)", -40.0, 30.0, 0.0, 0.1),
            'AG': st.slider("阴离子间隙 (mmol/L)", 5.0, 50.0, 12.0, 0.1),
            'WBC': st.slider("白细胞计数 (×10⁹/L)", 0.0, 400.0, 8.0, 0.01),
            'LYMP%': st.slider("淋巴细胞百分比", 0.0, 99.0, 30.0, 0.01)
        }

    validate_inputs(inputs['PH'], inputs['PO2'], inputs['Lac'])
    input_df = pd.DataFrame([inputs])

    if st.button("开始风险评估", type="primary"):
        with st.status("分析中...", expanded=True) as status:
            try:
                st.write("🔍 数据验证中...")
                if input_df.isnull().any().any():
                    raise ValueError("输入数据包含无效值")

                st.write("🧠 进行模型预测...")
                risk = make_prediction(model, scaler, input_df)

                status.update(label="分析完成", state="complete")
                col1, col2 = st.columns([1, 2])  # 明确定义列对象

                with col1:
                    st.subheader("风险评估结果")
                    st.metric("危重症发生概率", f"{risk * 100:.1f}%")
                    st.progress(risk, text=f"风险等级：{min(int(risk * 10) + 1, 10)}级")

                with col2:
                    st.subheader("关键影响因素")
                    html_content = generate_shap_plot(model, scaler, input_df)
                    st_html(html_content, height=600, scrolling=False)

            except Exception as e:
                status.update(label="分析失败", state="error")
                st.error(f"错误: {str(e)}")


if __name__ == "__main__":
    main()