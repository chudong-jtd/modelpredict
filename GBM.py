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

# Type definitions
SklearnModel = RandomForestClassifier
SklearnScaler = StandardScaler

def validate_and_convert_to_float(input_value: str, input_name: str) -> float:
    """Validate and convert input string to float"""
    try:
        return float(input_value)
    except ValueError:
        st.error(f"Invalid input for {input_name}. Please enter a valid number.")
        return None

def load_model_and_scaler() -> Tuple[SklearnModel, SklearnScaler]:
    """Load model and scaler"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(current_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()

def validate_inputs(PH: float, PO2: float, Lac: float):
    """Validate clinical parameters"""
    alerts = []
    if PH < 7.2:
        alerts.append("Severe acidosis (pH < 7.2)")
    elif PH < 7.35:
        alerts.append("Mild acidosis (7.2 ≤ pH < 7.35)")
    elif PH > 7.45:
        alerts.append("Alkalosis (pH > 7.45)")

    if PO2 < 50:
        alerts.append("Severe hypoxia (PO₂ < 50 mmHg)")
    elif PO2 < 60:
        alerts.append("Mild hypoxia (50 ≤ PO₂ < 60 mmHg)")
    elif PO2 > 300:
        alerts.append("Abnormal hyperoxia (PO₂ > 300 mmHg)")

    if Lac > 4:
        alerts.append("Severe hyperlactatemia (Lac > 4 mmol/L)")
    elif Lac > 2:
        alerts.append("Hyperlactatemia (2 < Lac ≤ 4 mmol/L)")

    for alert in alerts:
        if "Severe" in alert:
            st.error(f"🚨 {alert}")
        elif "Abnormal" in alert:
            st.error(f"⚠ {alert}")
        else:
            st.warning(f"⚠ {alert}")

def make_prediction(model: SklearnModel, scaler: SklearnScaler, input_data: pd.DataFrame) -> float:
    """Perform prediction"""
    try:
        scaled_data = scaler.transform(input_data)
        proba = model.predict_proba(scaled_data)
        return proba[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()

def generate_shap_plot(model: SklearnModel, scaler: SklearnScaler, input_data: pd.DataFrame) -> str:
    """Generate optimized SHAP visualization"""
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
        page_title="Clinical Prediction System",
        layout="wide",
        page_icon="🏥"
    )
    st.title("🏥 Critical Illness AKI Mortality Risk Prediction System")

    with st.sidebar:
        st.header("Patient Physiological Parameters")
        inputs = {
            'SBP': validate_and_convert_to_float(st.text_input("Systolic Blood Pressure (mmHg)", value="120.0"), "Systolic Blood Pressure"),
            'UO': validate_and_convert_to_float(st.text_input("Urine Output (mL/24h)", value="1500.0"), "Urine Output"),
            'PLT': validate_and_convert_to_float(st.text_input("Platelets (×10⁹/L)", value="200.0"), "Platelets"),
            'Na': validate_and_convert_to_float(st.text_input("Serum Sodium (mmol/L)", value="140.0"), "Serum Sodium"),
            'LDH': validate_and_convert_to_float(st.text_input("Lactate Dehydrogenase (U/L)", value="200.0"), "Lactate Dehydrogenase"),
            'PH': validate_and_convert_to_float(st.text_input("Arterial Blood PH", value="7.4"), "Arterial Blood PH"),
            'PO2': validate_and_convert_to_float(st.text_input("Arterial Partial Pressure of Oxygen (mmHg)", value="95.0"), "Arterial Partial Pressure of Oxygen"),
            'Lac': validate_and_convert_to_float(st.text_input("Arterial Blood Lactate (mmol/L)", value="1.2"), "Arterial Blood Lactate"),
            'BE': validate_and_convert_to_float(st.text_input("Arterial Blood Base Excess (mmol/L)", value="0.0"), "Arterial Blood Base Excess"),
            'AG': validate_and_convert_to_float(st.text_input("Anion Gap (mmol/L)", value="12.0"), "Anion Gap"),
            'WBC': validate_and_convert_to_float(st.text_input("White Blood Cell Count (×10⁹/L)", value="8.0"), "White Blood Cell Count"),
            'LYMP%': validate_and_convert_to_float(st.text_input("Lymphocyte Percentage", value="30.0"), "Lymphocyte Percentage")
        }

    if any(value is None for value in inputs.values()):
        st.warning("Please correct the invalid inputs before proceeding.")
        return

    validate_inputs(inputs['PH'], inputs['PO2'], inputs['Lac'])
    input_df = pd.DataFrame([inputs])

    if st.button("Start Risk Assessment", type="primary"):
        with st.status("Analyzing...", expanded=True) as status:
            try:
                st.write("🔍 Validating data...")
                if input_df.isnull().any().any():
                    raise ValueError("Input data contains invalid values")

                st.write("🧠 Running model prediction...")
                risk = make_prediction(model, scaler, input_df)

                status.update(label="Analysis complete", state="complete")
                col1, col2 = st.columns([1, 2])  # Explicitly define column objects

                with col1:
                    st.subheader("Risk Assessment Result")
                    st.metric("Probability of Mortality", f"{risk * 100:.1f}%")
                    st.progress(risk, text=f"Risk Level: {min(int(risk * 10) + 1, 10)}")

                with col2:
                    st.subheader("关键影响因素")
                    html_content = generate_shap_plot(model, scaler, input_df)
                    st_html(html_content, height=600, scrolling=False)

            except Exception as e:
                status.update(label="分析失败", state="error")
                st.error(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
