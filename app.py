import streamlit as st
import pickle
import numpy as np

# 1. Page Title & Styling
st.set_page_config(page_title="Smart Card User Segmentation", layout="centered")
st.title("📂 Smart Card Project - Clustering App")
st.write("Enter the customer details below to predict their target cluster:")

# 2. Model aur Scaler load karo
@st.cache_resource
def load_assets():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()

    # 3. Input fields divided into 2 Columns for better UI
    st.subheader("📋 Customer Demographics & Details")
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", value=30, step=1)
        Income = st.number_input("Income", value=50000.0, step=500.0)
        Recency = st.number_input("Recency (Days since last purchase)", value=15, step=1)
        Customer_Tenure_Days = st.number_input("Customer Tenure (Days)", value=365, step=1)
        Total_Spending = st.number_input("Total Spending", value=500.0, step=10.0)
        Total_Children = st.number_input("Total Children", value=0, step=1)
        Complain = st.selectbox("Has Complained? (0 = No, 1 = Yes)", [0, 1])
        Response = st.selectbox("Accepted last campaign? (0 = No, 1 = Yes)", [0, 1])

    with col2:
        NumDealsPurchases = st.number_input("Num Deals Purchases", value=1, step=1)
        NumWebPurchases = st.number_input("Num Web Purchases", value=2, step=1)
        NumCatalogPurchases = st.number_input("Num Catalog Purchases", value=1, step=1)
        NumStorePurchases = st.number_input("Num Store Purchases", value=3, step=1)
        NumWebVisitsMonth = st.number_input("Num Web Visits Month", value=5, step=1)

    st.subheader("🎓 Education Level (Select One as 1, others as 0)")
    col3, col4, col5 = st.columns(3)
    with col3:
        Education_Graduate = st.selectbox("Education: Graduate", [1, 0])
    with col4:
        Education_Postgraduate = st.selectbox("Education: Postgraduate", [0, 1])
    with col5:
        Education_Undergraduate = st.selectbox("Education: Undergraduate", [0, 1])

    st.subheader("🏠 Living Status (Select One as 1, others as 0)")
    col6, col7 = st.columns(2)
    with col6:
        Living_With_Alone = st.selectbox("Living: Alone", [0, 1])
    with col7:
        Living_With_Partner = st.selectbox("Living: With Partner", [1, 0])

    # 4. Predict button
    st.markdown("---")
    if st.button("Predict Cluster", use_container_width=True):
        # Saare 18 features ko EXACT usi sequence me daalna hai jo aapne bheja hai
        input_data = np.array([[
            Income, Recency, NumDealsPurchases, NumWebPurchases, 
            NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, 
            Complain, Response, Age, Customer_Tenure_Days, Total_Spending, 
            Total_Children, Education_Graduate, Education_Postgraduate, 
            Education_Undergraduate, Living_With_Alone, Living_With_Partner
        ]])
        
        # Scale corporate data
        scaled_data = scaler.transform(input_data)
        
        # Prediction
        cluster = model.predict(scaled_data)[0]
        
        # Big Success Box
        st.balloons()
        st.success(f"🎉 **This customer belongs to Cluster Number: {cluster}**")

except FileNotFoundError as e:
    st.error(f"Error: Model (`kmeans_model.pkl`) or Scaler (`scaler.pkl`) file missing in GitHub repo.")
