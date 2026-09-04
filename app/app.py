import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #172554 100%
        );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        padding: 35px 40px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #1e293b,
            #1e3a8a
        );
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 15px;
        padding: 7px 14px;
        border-radius: 20px;
        background: rgba(255,255,255,0.10);
        color: #bfdbfe;
        font-size: 14px;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: white;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Cards */
    .info-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(30,41,59,0.78);
        border: 1px solid rgba(148,163,184,0.15);
        min-height: 115px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .card-label {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 7px;
    }

    .card-value {
        color: white;
        font-size: 26px;
        font-weight: 750;
    }

    /* Result cards */
    .result-card {
        padding: 28px;
        border-radius: 20px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.16);
        text-align: center;
        min-height: 170px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.20);
    }

    .result-icon {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .result-label {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Risk badges */
    .risk-low {
        display: inline-block;
        padding: 9px 18px;
        border-radius: 25px;
        background: rgba(34,197,94,0.16);
        color: #4ade80;
        font-weight: 800;
        font-size: 20px;
    }

    .risk-medium {
        display: inline-block;
        padding: 9px 18px;
        border-radius: 25px;
        background: rgba(245,158,11,0.16);
        color: #fbbf24;
        font-weight: 800;
        font-size: 20px;
    }

    .risk-high {
        display: inline-block;
        padding: 9px 18px;
        border-radius: 25px;
        background: rgba(239,68,68,0.16);
        color: #f87171;
        font-weight: 800;
        font-size: 20px;
    }

    /* Prediction banner */
    .prediction-stay {
        padding: 20px;
        border-radius: 18px;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.35);
        color: #86efac;
        font-size: 22px;
        font-weight: 750;
        text-align: center;
    }

    .prediction-churn {
        padding: 20px;
        border-radius: 18px;
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.35);
        color: #fca5a5;
        font-size: 22px;
        font-weight: 750;
        text-align: center;
    }

    /* Interpretation */
    .interpretation {
        padding: 22px;
        border-radius: 18px;
        background: rgba(30,41,59,0.75);
        border-left: 5px solid #60a5fa;
        color: #dbeafe;
        font-size: 16px;
        line-height: 1.7;
    }

    /* Model cards */
    .model-card {
        text-align: center;
        padding: 20px;
        border-radius: 18px;
        background: rgba(30,41,59,0.75);
        border: 1px solid rgba(148,163,184,0.14);
    }

    .model-number {
        font-size: 28px;
        font-weight: 800;
        color: #60a5fa;
    }

    .model-name {
        color: #cbd5e1;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        min-height: 55px;
        border-radius: 14px;
        border: none;
        font-size: 17px;
        font-weight: 750;
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );
        color: white;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(37,99,235,0.35);
    }

    /* Input labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 35px;
        margin-top: 40px;
        border-top: 1px solid rgba(148,163,184,0.12);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "customer_churn_model.pkl"

)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.error(f"Error: {e}")
    st.error(f"Model path: {MODEL_PATH}")
    st.stop()


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔮 ChurnAI</div>

        <div class="hero-subtitle">
            Intelligent customer churn prediction powered by
            machine learning.
        </div>

        <div class="hero-badge">
            MACHINE LEARNING • CUSTOMER ANALYTICS • RISK PREDICTION
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL CHECK
# =========================================================

if not model_loaded:

    st.error("The trained model could not be loaded.")

    st.code(model_error)

    st.stop()


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the customer characteristics below to estimate their '
    'churn probability.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PERSONAL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        '<div class="card-label">PERSONAL INFORMATION</div>',
        unsafe_allow_html=True
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )


# =========================================================
# TELECOMMUNICATION SERVICES
# =========================================================

with col2:

    st.markdown(
        '<div class="card-label">TELECOMMUNICATION SERVICES</div>',
        unsafe_allow_html=True
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# =========================================================
# ADDITIONAL SERVICES
# =========================================================

with col3:

    st.markdown(
        '<div class="card-label">ADDITIONAL SERVICES</div>',
        unsafe_allow_html=True
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# =========================================================
# CONTRACT & BILLING
# =========================================================

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

billing_col1, billing_col2, billing_col3 = st.columns(3)


with billing_col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with billing_col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with billing_col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=18.25,
        max_value=118.75,
        value=65.00,
        step=0.50
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=8684.80,
        value=780.0,
        step=10.0
    )


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🚀 ANALYZE CUSTOMER CHURN RISK",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # Prediction
        prediction = model.predict(
            customer_data
        )[0]

        # Probability
        probabilities = model.predict_proba(
            customer_data
        )[0]

        # Safely identify the Yes class
        class_names = list(model.classes_)

        yes_index = class_names.index("Yes")

        probability = probabilities[yes_index]

        probability_percent = probability * 100


        # =================================================
        # RISK CLASSIFICATION
        # =================================================

        if probability_percent >= 70:

            risk_level = "HIGH RISK"
            risk_class = "risk-high"
            risk_icon = "🔴"

        elif probability_percent >= 40:

            risk_level = "MEDIUM RISK"
            risk_class = "risk-medium"
            risk_icon = "🟠"

        else:

            risk_level = "LOW RISK"
            risk_class = "risk-low"
            risk_icon = "🟢"


        # =================================================
        # RESULTS
        # =================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2, result_col3 = st.columns(3)


        # Prediction
        with result_col1:

            if prediction == "Yes":

                st.markdown(
                    """
                    <div class="result-card">
                        <div class="result-icon">⚠️</div>
                        <div class="result-label">
                            Prediction
                        </div>
                        <div class="result-value">
                            Likely to Churn
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-card">
                        <div class="result-icon">✅</div>
                        <div class="result-label">
                            Prediction
                        </div>
                        <div class="result-value">
                            Likely to Stay
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # Probability
        with result_col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-icon">📈</div>
                    <div class="result-label">
                        Churn Probability
                    </div>
                    <div class="result-value">
                        {probability_percent:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Risk
        with result_col3:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-icon">
                        {risk_icon}
                    </div>

                    <div class="result-label">
                        Risk Level
                    </div>

                    <div style="margin-top:12px;">
                        <span class="{risk_class}">
                            {risk_level}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # Probability progress bar
        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(
            float(probability)
        )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.markdown(
            '<div class="section-title">🧠 Model Interpretation</div>',
            unsafe_allow_html=True
        )


        if probability_percent >= 70:

            interpretation = (
                "This customer has a high predicted probability "
                "of churn. Proactive retention strategies and "
                "targeted engagement may be appropriate."
            )

        elif probability_percent >= 40:

            interpretation = (
                "This customer has a moderate predicted churn "
                "risk. Monitoring and targeted engagement may "
                "be appropriate."
            )

        else:

            interpretation = (
                "This customer has a relatively low predicted "
                "probability of churn."
            )


        st.markdown(
            f"""
            <div class="interpretation">
                {interpretation}
            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # MODEL PERFORMANCE
        # =================================================

        st.markdown(
            '<div class="section-title">🤖 Model Performance</div>',
            unsafe_allow_html=True
        )

        perf1, perf2, perf3 = st.columns(3)


        with perf1:

            st.markdown(
                """
                <div class="model-card">
                    <div class="model-number">84.25%</div>
                    <div class="model-name">
                        ROC-AUC
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with perf2:

            st.markdown(
                """
                <div class="model-card">
                    <div class="model-number">75.13%</div>
                    <div class="model-name">
                        Churn Recall
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with perf3:

            st.markdown(
                """
                <div class="model-card">
                    <div class="model-number">63.57%</div>
                    <div class="model-name">
                        Churn F1 Score
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # CUSTOMER SUMMARY
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("👁️ View Customer Information"):

            st.dataframe(
                customer_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction failed. Please check the input values "
            "and model configuration."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🔮 ChurnAI &nbsp;•&nbsp;
        Customer Churn Prediction System
        <br><br>
        Built with Python • Scikit-learn • Pandas • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
