import streamlit as st
import pandas as pd
from joblib import load
import dill


with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dic = load('my_feature_dict.pkl')


def predict_churn(data):
    prediction = model.predict(data)
    return prediction

st.title("Customer Churn Prediction APP")
st.subheader("Based on Telecom Dataset")

z = st.slider('Select a value: ', 0, 100, 1)
st.write(f"The square of {z} is {z**2}")