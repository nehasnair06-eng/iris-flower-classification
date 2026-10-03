import joblib
import numpy as np
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Iris AI", page_icon="🌸", layout="centered")


# =========================================================
# LOAD MODEL & TARGETS
# =========================================================


@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model/iris_model.pkl")
        target_names = joblib.load("model/target_names.pkl")
        return model, target_names
    except Exception as e:
        st.error(
            f"Error loading model files: {e}. Ensure 'model/iris_model.pkl' and 'model/target_names.pkl' exist."
        )
        st.stop()


model, target_names = load_assets()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #fff0f6 0%,
        #ffe4ef 45%,
        #fff7fa 100%
    );
}

/* Hide Streamlit elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main container */
.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 25px 20px 15px 20px;
}

.flower {
    font-size: 65px;
    margin-bottom: 5px;
}

.title {
    font-size: 45px;
    font-weight: 700;
    color: #b83268;
    margin: 0;
}

.subtitle {
    color: #8f526c;
    font-size: 16px;
    margin-top: 8px;
}

/* Info cards */
.info-container {
    display: flex;
    gap: 15px;
    margin: 25px 0;
}

.info-card {
    flex: 1;
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(255,182,193,0.5);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(184,50,104,0.08);
}

.info-title {
    color: #b83268;
    font-size: 13px;
    font-weight: 600;
}

.info-value {
    color: #5c3042;
    font-size: 15px;
    font-weight: 500;
    margin-top: 5px;
}

/* Section title */
.section-title {
    color: #a52d60;
    font-size: 21px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 10px;
}

/* Input labels */
label {
    color: #71364d !important;
    font-weight: 500 !important;
}

/* Number inputs */
.stNumberInput input {
    border: 1.5px solid #f3b6cc !important;
    border-radius: 12px !important;
    background-color: rgba(255,255,255,0.85) !important;
    color: #5c3042 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 13px;
    background: linear-gradient(
        90deg,
        #d94f83,
        #b83268
    );
    color: white;
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(184,50,104,0.25);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(184,50,104,0.35);
}

/* Result card */
.result-card {
    margin-top: 28px;
    padding: 30px;
    border-radius: 22px;
    background: rgba(255,255,255,0.85);
    border: 1px solid #f3bfd1;
    text-align: center;
    box-shadow: 0 12px 35px rgba(184,50,104,0.12);
}

.result-label {
    color: #9b5872;
    font-size: 14px;
}

.result-flower {
    font-size: 42px;
    margin: 5px;
}

.result-name {
    color: #b83268;
    font-size: 30px;
    font-weight: 700;
    text-transform: capitalize;
}

.confidence {
    color: #6e4655;
    font-size: 15px;
    margin-top: 8px;
}

/* Footer */
.custom-footer {
    text-align: center;
    color: #a56b80;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
    <div class="flower">🌸</div>
    <div class="title">Iris AI</div>
    <div class="subtitle">Intelligent Iris Flower Classification</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# INFO CARDS
# =========================================================

st.markdown(
    """
<div class="info-container">
    <div class="info-card">
        <div class="info-title">MODEL</div>
        <div class="info-value">Random Forest</div>
    </div>
    <div class="info-card">
        <div class="info-title">TASK</div>
        <div class="info-value">Classification</div>
    </div>
    <div class="info-card">
        <div class="info-title">PLATFORM</div>
        <div class="info-value">Streamlit</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">🌿 Flower Measurements</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1
    )
    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1
    )
    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")
predict = st.button("🌸 Predict Flower")


# =========================================================
# PREDICTION & DISPLAY
# =========================================================

if predict:
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )

    prediction = model.predict(input_data)
    predicted_class = target_names[prediction[0]]
    probabilities = model.predict_proba(input_data)
    confidence = np.max(probabilities) * 100

    st.markdown(
        f"""
    <div class="result-card">
        <div class="result-label">Predicted Species</div>
        <div class="result-flower">🌸</div>
        <div class="result-name">{predicted_class}</div>
        <div class="confidence">Confidence: <strong>{confidence:.2f}%</strong></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="custom-footer">Powered by Streamlit & Scikit-Learn</div>',
    unsafe_allow_html=True,
)