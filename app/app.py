import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "customer_churn_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .risk-high {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #d9534f;
    }

    .risk-low {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #5cb85c;
    }

    .metric-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Powered Customer Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# MODEL CHECK
# =========================================================

if not model_loaded:

    st.error(
        "The trained model could not be loaded."
    )

    st.code(model_error)

    st.stop()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

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


with col2:

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


with col3:

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
# CONTRACT AND BILLING
# =========================================================

st.subheader("💳 Contract and Billing")

col4, col5, col6 = st.columns(3)


with col4:

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


with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col6:

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


st.divider()


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
# PREDICTION
# =========================================================

st.subheader("🔮 Churn Prediction")

predict_button = st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    try:

        prediction = model.predict(
            customer_data
        )[0]

        probability = model.predict_proba(
            customer_data
        )[0, 1]

        probability_percent = probability * 100


        # =================================================
        # RISK CLASSIFICATION
        # =================================================

        if probability_percent >= 70:

            risk_level = "HIGH RISK"
            risk_class = "risk-high"

        elif probability_percent >= 40:

            risk_level = "MEDIUM RISK"
            risk_class = "risk-high"

        else:

            risk_level = "LOW RISK"
            risk_class = "risk-low"


        st.divider()

        st.subheader("📈 Prediction Result")


        # =================================================
        # RESULT COLUMNS
        # =================================================

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            if prediction == "Yes":

                st.error(
                    "⚠️ CUSTOMER LIKELY TO CHURN"
                )

            else:

                st.success(
                    "✅ CUSTOMER LIKELY TO STAY"
                )


        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.1f}%"
            )


        with result_col3:

            st.metric(
                "Risk Level",
                risk_level
            )


        st.progress(
            float(probability)
        )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader("🧠 Model Interpretation")

        if probability_percent >= 70:

            st.warning(
                "This customer has a high predicted "
                "probability of churn. Consider proactive "
                "customer-retention strategies."
            )

        elif probability_percent >= 40:

            st.info(
                "This customer has a moderate predicted "
                "churn risk. Monitoring and targeted "
                "engagement may be appropriate."
            )

        else:

            st.success(
                "This customer has a relatively low "
                "predicted probability of churn."
            )


        # =================================================
        # CUSTOMER SUMMARY
        # =================================================

        with st.expander(
            "View Customer Information"
        ):

            st.dataframe(
                customer_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)
