
import streamlit as st
import joblib
import pandas as pd

from url_features import extract_url_features


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔐",
    layout="centered"
)


# --------------------------------------------------
# 2. Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    color: #172554;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}

.stButton > button {
    width: 100%;
    background-color: #1d4ed8;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    artifact = joblib.load("url_phishing_model.pkl")

    return artifact


try:
    artifact = load_model()
    model = artifact["pipeline"]

except FileNotFoundError:
    st.error(
        "Model file not found. Run train_model.py first."
    )
    st.stop()


# --------------------------------------------------
# 4. Application heading
# --------------------------------------------------

st.markdown(
    '<div class="title">Phishing URL Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect suspicious URLs using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# --------------------------------------------------
# 5. URL input
# --------------------------------------------------

url = st.text_input(
    "Enter the URL",
    placeholder="https://example.com"
)


# --------------------------------------------------
# 6. Prediction
# --------------------------------------------------

if st.button("Check URL"):

    if not url.strip():

        st.warning("Please enter a URL.")

    else:

        try:

            # Extract features
            features = extract_url_features(url)

            # Convert dictionary into DataFrame
            input_df = pd.DataFrame([features])

            # Predict class
            prediction = model.predict(input_df)[0]

            # Predict probability
            probabilities = model.predict_proba(input_df)[0]

            # Class 0 = Phishing
            # Class 1 = Legitimate
            phishing_probability = probabilities[
                list(model.classes_).index(0)
            ]

            legitimate_probability = probabilities[
                list(model.classes_).index(1)
            ]

            st.subheader("Analysis Result")

            if prediction == 0:

                st.error("Warning: This URL is predicted as PHISHING.")

            else:

                st.success("This URL is predicted as LEGITIMATE.")

            st.write(
                f"Phishing probability: "
                f"{phishing_probability * 100:.2f}%"
            )

            st.write(
                f"Legitimate probability: "
                f"{legitimate_probability * 100:.2f}%"
            )

            st.progress(float(phishing_probability))

            with st.expander("View extracted URL features"):

                st.dataframe(input_df.T.rename(
                    columns={0: "Extracted value"}
                ))

        except Exception as e:

            st.error(f"Unable to analyse this URL: {e}")


# --------------------------------------------------
# 7. Disclaimer
# --------------------------------------------------

st.caption(
    "This is an ML-based prediction, not a guarantee "
    "that a website is safe. Do not enter passwords or "
    "sensitive information on suspicious websites."
)
