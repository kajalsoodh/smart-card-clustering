import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="SmartCart Clustering", layout="centered")

st.title("📂 Smart Card Project - Clustering App")
st.write("Enter the customer details below to predict their target cluster:")

# 1. Assets load karo
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
    # 2. User Inputs Form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        income = st.number_input("Income", min_value=0, value=50000)
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=25)
        tenure = st.number_input("Customer Tenure Days", min_value=0, value=500)
        total_spending = st.number_input("Total Spending", min_value=0, value=1000)
        total_children = st.number_input("Total Children", min_value=0, max_value=10, value=1)

    with col2:
        num_web_visits = st.number_input("Num Web Visits Month", min_value=0, value=5)
        num_deals = st.number_input("Num Deals Purchases", min_value=0, value=2)
        num_web_purchases = st.number_input("Num Web Purchases", min_value=0, value=3)
        num_catalog = st.number_input("Num Catalog Purchases", min_value=0, value=2)
        num_store = st.number_input("Num Store Purchases", min_value=0, value=5)
        complain = st.selectbox("Complain (0 = No, 1 = Yes)", [0, 1])

    # Dropdowns for Categorical Data
    st.markdown("---")
    education = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate", "Other"])
    living_status = st.selectbox("Living Status", ["Alone", "Partner", "Other"])
    response = st.selectbox("Accepted Campaign Response (0 = No, 1 = Yes)", [0, 1])

    # Dummy variables encoding logic
    edu_graduate = 1 if education == "Graduate" else 0
    edu_postgraduate = 1 if education == "Postgraduate" else 0
    edu_undergraduate = 1 if education == "Undergraduate" else 0
    
    living_alone = 1 if living_status == "Alone" else 0
    living_partner = 1 if living_status == "Partner" else 0

    # 3. Predict Button
    if st.button("Predict Cluster", use_container_width=True):
        # Dataframe exact usi sequence me jisme training hui thi
        input_df = pd.DataFrame([{
            'Income': income,
            'Recency': recency,
            'NumDealsPurchases': num_deals,
            'NumWebPurchases': num_web_purchases,
            'NumCatalogPurchases': num_catalog,
            'NumStorePurchases': num_store,
            'NumWebVisitsMonth': num_web_visits,
            'Complain': complain,
            'Response': response,
            'Age': age,
            'Customer_Tenure_Days': tenure,
            'Total_Spending': total_spending,
            'Total_Children': total_children,
            'Education_Graduate': edu_graduate,
            'Education_Postgraduate': edu_postgraduate,
            'Education_Undergraduate': edu_undergraduate,
            'Living_With_Alone': living_alone,
            'Living_With_Partner': living_partner
        }])
        
        try:
            # Scale -> PCA -> Predict
            scaled_data = scaler.transform(input_df)
            pca_data = pca.transform(scaled_data)
            cluster = model.predict(pca_data)[0]
            
            st.balloons()
            st.success(f"🎉 Target Cluster: {cluster}")
        except Exception as e:
            st.error(f"Internal Error: {e}")
