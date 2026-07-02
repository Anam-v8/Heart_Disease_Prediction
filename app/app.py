from utils import preprocess_input
import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# LOAD CSS
# ============================================

def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

# ============================================
# LOAD MODEL
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.title("❤️ Heart Predictor")

    st.markdown("---")

    st.subheader("About")

    st.write("""
Predict the likelihood of heart disease using a
Machine Learning model trained on clinical data.
""")

    st.markdown("---")

    st.subheader("Model")

    st.success("Random Forest")

    st.write("Accuracy : **83.6%**")

    st.write("Features : **13**")

    st.write("Dataset : Heart Disease")

    st.markdown("---")

    st.subheader("Developer")

    st.write("👩 Anamika")

    st.write("Python • Machine Learning")

    st.markdown("---")

    st.caption(
        "This application is for educational purposes only."
    )

# ============================================
# HEADER
# ============================================

st.title("❤️ Heart Disease Prediction System")

st.markdown("""
Predict whether a patient is at risk of heart disease using
Machine Learning.

Fill in the patient's information and click **Predict**.
""")

st.divider()

st.subheader("🩺 Patient Information")

col1, col2 = st.columns(2)

# ================= LEFT ====================

with col1:

    age = st.number_input(
        "Age",
        1,
        120,
        45,
        help="Patient age in years"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    chol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    restecg = st.selectbox(
        "Rest ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    slope = st.selectbox(
        "Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )

# ================= RIGHT ====================

with col2:

    sex = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        80,
        250,
        120
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar >120 mg/dl",
        [
            "No",
            "Yes"
        ]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0,
        step=0.1
    )

    ca = st.selectbox(
        "Major Vessels",
        [0,1,2,3,4]
    )

st.divider()

# ============================================
# PREDICTION
# ============================================

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    # Prepare data
    features = preprocess_input(
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    )

    # Scale
    features_scaled = scaler.transform(features.values)

    # Prediction
    prediction = model.predict(features_scaled)[0]

    probability = model.predict_proba(features_scaled)[0]

    st.divider()

    left, right = st.columns([1, 1])

    # ===========================
    # RESULT CARD
    # ===========================

    with left:

        st.subheader("❤️ Prediction")

        if prediction == 1:

            confidence = probability[1] * 100

            st.success("🟢 LOW RISK")

        else:

            confidence = probability[0] * 100

            st.error("🔴 HIGH RISK")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

    # ===========================
    # PATIENT SUMMARY
    # ===========================

    with right:

        st.subheader("📋 Patient Summary")

        summary = pd.DataFrame({

            "Feature":[
                "Age",
                "Gender",
                "Chest Pain",
                "Blood Pressure",
                "Cholesterol",
                "Fasting Sugar",
                "Rest ECG",
                "Max Heart Rate",
                "Exercise Angina",
                "Old Peak",
                "Slope",
                "Major Vessels",
                "Thalassemia"
            ],

            "Value":[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ===========================
    # RECOMMENDATIONS
    # ===========================

    st.subheader("💡 Recommendations")

    if prediction == 1:

        col1, col2 = st.columns(2)

        with col1:

            st.success("""
✅ Maintain a balanced diet

✅ Exercise regularly

✅ Stay hydrated

✅ Maintain healthy weight
""")

        with col2:

            st.success("""
✅ Annual health check-up

✅ Reduce stress

✅ Sleep 7-8 hours

✅ Avoid smoking
""")

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.warning("""
⚠ Consult a cardiologist

⚠ Monitor blood pressure

⚠ Reduce cholesterol

⚠ Follow prescribed medicines
""")

        with col2:

            st.warning("""
⚠ Healthy low-fat diet

⚠ Regular medical check-ups

⚠ Avoid smoking

⚠ Exercise only if advised
""")

    st.divider()

    st.info(
        "This prediction is generated by a Machine Learning model and should not replace professional medical diagnosis."
    )

# ============================================
# FEATURE IMPORTANCE
# ============================================

st.markdown("---")

st.subheader("📊 Feature Importance")

feature_names = [
    "Age",
    "Sex",
    "Chest Pain",
    "Rest BP",
    "Cholesterol",
    "Fasting Sugar",
    "Rest ECG",
    "Max Heart Rate",
    "Exercise Angina",
    "Old Peak",
    "Slope",
    "Major Vessels",
    "Thal"
]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

with st.expander("📈 Show Feature Importance Chart"):

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.barh(
        importance["Feature"],
        importance["Importance"],
        color="#3B82F6"
    )

    ax.invert_yaxis()

    ax.set_xlabel("Importance Score")

    ax.set_ylabel("")

    ax.set_title("Random Forest Feature Importance")

    plt.tight_layout()

    st.pyplot(fig)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center; color:gray; font-size:14px;'>

❤️ <b>Heart Disease Prediction System</b><br>

Developed using <b>Python, Scikit-Learn and Streamlit</b>

</div>
""",
    unsafe_allow_html=True
)