{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea8795ab-0421-4064-b957-abc5d506f137",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# 1. Title aur Description\n",
    "st.title(\"📂 Customer Segmentation / Clustering App\")\n",
    "st.write(\"Apne features ki values enter karo aur dekho ye kis cluster mein aata hai.\")\n",
    "\n",
    "# 2. Saved Model aur Scaler ko load karo\n",
    "@st.cache_resource # Taaki baar-baar load na ho\n",
    "def load_assets():\n",
    "    with open('kmeans_model.pkl', 'rb') as f:\n",
    "        model = pickle.load(f)\n",
    "    with open('scaler.pkl', 'rb') as f:\n",
    "        scaler = pickle.load(f)\n",
    "    return model, scaler\n",
    "\n",
    "model, scaler = load_assets()\n",
    "\n",
    "# 3. User Input Fields (Example ke liye maine 2 features liye hain, aap apne data ke hisab se badal lena)\n",
    "st.subheader(\"Enter Feature Values:\")\n",
    "feature_1 = st.number_input(\"Feature 1 Value (e.g., Age)\", value=25.0)\n",
    "feature_2 = st.number_input(\"Feature 2 Value (e.g., Income)\", value=50000.0)\n",
    "\n",
    "# 4. Prediction Logic\n",
    "if st.button(\"Predict Cluster\"):\n",
    "    # Input ko array mein convert karo\n",
    "    input_data = np.array([[feature_1, feature_2]])\n",
    "    \n",
    "    # NEW DATA KO BHI SCALE KARNA ZAROORI HAI!\n",
    "    scaled_data = scaler.transform(input_data)\n",
    "    \n",
    "    # Predict cluster\n",
    "    cluster = model.predict(scaled_data)[0]\n",
    "    \n",
    "    # Output dikhao\n",
    "    st.success(f\"🎉 Yeh data **Cluster Number: {cluster}** se belong karta hai!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
