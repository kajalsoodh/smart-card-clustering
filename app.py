import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("📂 Smart Card Project - Clustering App")
st.write("Enter the customer details below to predict their target cluster:")

@st.cache_resource
def load_assets():
    try:
        scaler = pickle.load(open('smartcart_scaler.pkl', 'rb'))
        pca = pickle.load(open('smartcart_pca.pkl', 'rb'))
        model = pickle.load(open('smartcart_model.pkl', 'rb'))
        return scaler, pca, model
    except FileNotFoundError:
        st.error("❌ Error: Model, Scaler, or PCA file missing in GitHub repo.")
        return None, None, None

scaler, pca, model = load_assets()

if scaler is not None:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    complain = st.selectbox("Complain (0 = No, 1 = Yes)", [0, 1])
    tenure = st.number_input("Customer Tenure Days", min_value=0, value=100)
    
    education = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Other"])
    edu_graduate = 1 if education == "Graduate" else 0
    edu_postgraduate = 1 if education == "Postgraduate" else 0

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([{
            'Age': age,
            'Complain': complain,
            'Customer_Tenure_Days': tenure,
            'Education_Graduate': edu_graduate,
            'Education_Postgraduate': edu_postgraduate
        }])
        
        try:
            scaled_data = scaler.transform(input_df)
            pca_data = pca.transform(scaled_data)
            cluster = model.predict(pca_data)[0]
            st.success(f"🎉 Target Cluster: {cluster}")
        except Exception as e:
            st.error(f"Internal Error: {e}")