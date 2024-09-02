import streamlit as st
import pandas as pd
import pickle
from joblib import load
from checking import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    # model1 = load(r"D:\ds_intern\iso_forest.pkl")
    # model2 = load(r"D:\ds_intern\lof.pkl")
    model1 = load(r"iso_forest.pkl")
    model2 = load(r"lof.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

# Set website title
st.set_page_config(page_title="ML Model Runner", layout="wide")

# Title of the website
st.title("Run ML Model on Your Dataset")

# Sidebar with navigation
st.sidebar.title("Choose your model")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Iso Forest", "Local Outlier Factor"])

# Home Page
if page == "Home":
    st.header("Home Page")
    st.write("""
    Welcome to the ML Model Runner website.
    
    Here you can upload your dataset, select a pre-trained model, and run predictions.
    """)

# Run Model Page
elif page == "Iso Forest":
    st.header("Run Your ML Model")

    # File uploader for user dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        try:

            # Button to make predictions
            if st.button("Run Model"):
                # Ensure that the input data matches the model's expected format
                predictions = detect_fraudulent_transactions(df,model1)
                
                # Display predictions
                st.write("Model Predictions:")
                st.dataframe(predictions)
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Run Model Page
elif page == "Local Outlier Factor":
    st.header("Run Your ML Model")

    # File uploader for user dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        try:

            # Button to make predictions
            if st.button("Run Model"):
                # Ensure that the input data matches the model's expected format
                predictions = detect_fraudulent_transactions(df,model2)
                
                # Display predictions
                st.write("Model Predictions:")
                st.dataframe(predictions)
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Footer
st.markdown("---")
st.write("Powered by [Streamlit](https://streamlit.io)")
