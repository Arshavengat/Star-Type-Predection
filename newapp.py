import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# Background setup
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_from_local("star.jpeg")

# Load model and encoder
with open("final_model_2.pkl", "rb") as file:
    model = pickle.load(file)
with open("star_color_encoder.pkl", "rb") as file:
    le = pickle.load(file)

# Custom CSS
st.markdown(
    """
    <style>
    .title-text {
        color: #00FFFF;
        font-size: 64px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 40px;
    }
    .stNumberInput label {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 600;
    }
    .stNumberInput input {
        font-size: 20px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="title-text">⭐ Spectral Class Prediction App</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.write("""
This app predicts the spectral class of a star (A, B, F, G, K, M, O) 
using stellar features and a trained machine learning model.
""")

# Inputs
temperature = st.number_input("🌡️ Temperature (K)", min_value=2000, max_value=40000, step=1000)
luminosity = st.number_input("💡 Luminosity (L/Lo)", min_value=0.0, step=0.1, format="%.2f")
radius = st.number_input("🪐 Radius (R/Ro)", min_value=0.0, step=0.1, format="%.2f")
magnitude = st.number_input("✨ Absolute magnitude (Mv)", step=0.1, format="%.2f")
star_type = st.number_input("Star type", min_value=0, max_value=6, step=1)

star_color = st.selectbox("Star color", [
    'Red', 'Blue White', 'White', 'Yellowish White', 'Pale yellow orange',
    'Blue', 'Whitish', 'Yellow-White', 'Orange', 'White-Yellow',
    'Yellowish', 'Orange-Red', 'Blue-White'
])

# Create DataFrame
user_inputs = {
    "Temperature (K)": temperature,
    "Luminosity(L/Lo)": luminosity,
    "Radius(R/Ro)": radius,
    "Absolute magnitude(Mv)": magnitude,
    "Star type": star_type,
    "Star color": star_color
}
user_df = pd.DataFrame([user_inputs])
user_df["Star color"] = le.transform(user_df["Star color"])

# Prediction
if st.button('🔮 Predict Spectral Class'):
    prediction = model.predict(user_df)
    classes = ["A", "B", "F", "G", "K", "M", "O"]
    st.success(f"🌟 Spectral Class: {classes[prediction[0]]}")
