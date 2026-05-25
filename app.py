import streamlit as st
import pickle
import numpy as np

# 1. Title
st.title("📂 Smart Card Project - Clustering App")
st.write("Enter values to predict the target cluster:")

# 2. Model aur Scaler load karo
@st.cache_resource
def load_assets():
    # File ke sahi naam check kar lena jo aapne upload kiye hain
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()

    # 3. Input fields (Aapke data columns ke hisab se change kar sakte ho)
    feat1 = st.number_input("Feature 1 (e.g., Feature 1)", value=0.0)
    feat2 = st.number_input("Feature 2 (e.g., Feature 2)", value=0.0)

    # 4. Predict button
    if st.button("Predict Cluster"):
        input_data = np.array([[feat1, feat2]])
        scaled_data = scaler.transform(input_data)
        cluster = model.predict(scaled_data)[0]
        st.success(f"🎉 Target Cluster: {cluster}")

except FileNotFoundError as e:
    st.error(f"Error: Model ya Scaler file nahi mili. Please check file names. Details: {e}")
