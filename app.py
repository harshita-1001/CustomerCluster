# Importing libraries
import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st



#load the saved model
kmeans = joblib.load("Model.pk1")
df=pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)" , "Spending Score (1-100)"]]
X_array = X.values

#Streamlit Application Page
st.set_page_config(page_title = "Customer Cluster Prediction" , layout = "centered")
st.title("Customer cluster Prediction ")
st.write("Enter the customer Annual Income and spending score to predict the cluster")

#inputs
annual_income = st.number_input("Annual Income of a Customer" , min_value = 0 , max_value = 400 , value = 50 )
spending_score = st.slider("Spending Scote  between 1-100" , 1 , 100 , 20)

#predict the cluster
if st.button("Predict Cluster"):
    input_data = np.array([[annual_income , spending_score]])
    cluster = kmeans.predict(input_data)
    st.success(f"Predicted Cluster is:{cluster}")
