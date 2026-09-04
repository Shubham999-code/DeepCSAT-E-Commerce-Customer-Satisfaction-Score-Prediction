import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(page_title="CSAT Predictor", page_icon="⭐", layout="centered")

st.title("e-Commerce CSAT Predictor")
st.write("Predict Customer Satisfaction Scores based on interaction details.")

# Load Model and Preprocessor
@st.cache_resource
def load_assets():
    try:
        model = load_model('csat_ann_model.h5')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.info("Make sure 'csat_ann_model.h5' and 'preprocessor.pkl' are in the same directory.")
        return None, None

model, preprocessor = load_assets()

if model is not None and preprocessor is not None:
    st.sidebar.header("Input Features")
    
    # Define Inputs
    channel_name = st.sidebar.selectbox("Channel", ["Outcall", "Inbound", "Email"])
    category = st.sidebar.selectbox("Category", ["Product Queries", "Order Related", "Returns", "Cancellation"])
    sub_category = st.sidebar.selectbox("Sub-Category", ["Life Insurance", "Installation/demo", "Reverse Pickup Enquiry", "Not Needed", "Fraudulent User", "Exchange / Replacement", "Missing", "Product Specific Information"])
    agent_shift = st.sidebar.selectbox("Agent Shift", ["Morning", "Evening", "Split", "Night"])
    tenure_bucket = st.sidebar.selectbox("Tenure Bucket", ["On Job Training", "0-30", "31-60", "61-90", ">90"])
    
    item_price = st.sidebar.number_input("Item Price ($)", min_value=0.0, value=50.0, step=1.0)
    connected_handling_time = st.sidebar.number_input("Handling Time (mins)", min_value=0.0, value=5.0, step=0.5)
    response_time_mins = st.sidebar.number_input("Response Time (mins)", min_value=0.0, value=15.0, step=1.0)
    has_remarks = st.sidebar.radio("Did customer leave remarks?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'channel_name': [channel_name],
        'category': [category],
        'Sub-category': [sub_category],
        'Item_price': [item_price],
        'connected_handling_time': [connected_handling_time],
        'Tenure Bucket': [tenure_bucket],
        'Agent Shift': [agent_shift],
        'has_remarks': [has_remarks],
        'response_time_mins': [response_time_mins]
    })
    
    st.subheader("Customer Interaction Summary")
    st.dataframe(input_data)
    
    if st.button("Predict CSAT"):
        with st.spinner("Predicting..."):
            try:
                # Preprocess
                X_processed = preprocessor.transform(input_data)
                
                # Predict
                y_pred_prob = model.predict(X_processed)
                y_pred = np.argmax(y_pred_prob, axis=1)[0]
                
                # Output result (Add 1 because model predicts 0-4)
                st.success(f"### Predicted CSAT Score: {y_pred + 1} ⭐")
                
                # Confidence
                confidence = np.max(y_pred_prob) * 100
                st.write(f"**Model Confidence:** {confidence:.2f}%")
                
                st.bar_chart(pd.DataFrame(y_pred_prob[0], index=["1⭐", "2⭐", "3⭐", "4⭐", "5⭐"], columns=["Probability"]))
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
else:
    st.warning("Model and preprocessor not loaded. Please ensure the Google Colab notebook has been executed to generate the required files.")
