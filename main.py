import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import sklearn.compose._column_transformer
import sklearn.impute

# FIX 1: Mock missing class for ColumnTransformer version mismatch
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# FIX 2: Dynamic attribute patch for SimpleImputer version mismatch & NumPy can_cast error
original_transform = sklearn.impute.SimpleImputer.transform
def patched_transform(self, X, *args, **kwargs):
    if not hasattr(self, '_fill_dtype'):
        # Extract the default fill value (usually 0)
        fallback_val = getattr(self, 'fill_value', 0)
        if fallback_val is None:
            fallback_val = 0
        # CRITICAL FIX: Convert the python int/float into a proper numpy dtype object 
        # to satisfy the new NumPy can_cast rules (NEP 50)
        self._fill_dtype = np.array(fallback_val).dtype
    return original_transform(self, X, *args, **kwargs)
sklearn.impute.SimpleImputer.transform = patched_transform


# 1. Load the pretrained pipeline using dill (exactly how it was saved)
with open('pipeline_class.pkl', 'rb') as file:
    model = dill.load(file)

# 2. Load the feature dictionary using pickle (exactly how it was saved)
with open('class_feature_dict.pkl', 'rb') as file:
    my_feature_dict = pickle.load(file)

# Function to predict employee churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# App Title and Subheader customized for Employee Churn
st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Domain Dataset')

# --- Categorical Features Section ---
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}

# Safely extract and loop over the column names and unique values from the dictionary structure
column_names = list(categorical_input.get('Column Name').values())
members_list = list(categorical_input.get('Members').values())

for i, col in enumerate(column_names):
    categorical_input_vals[col] = st.selectbox(col, members_list[i], key=col)

# --- Numerical Features Section ---
st.subheader('Numerical Features')
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}

for col in numerical_input.get('Column Name'):
    # Using integer input since values like Age, JoiningYear, and Experience are discrete integers
    numerical_input_vals[col] = st.number_input(col, step=1, value=0, key=col)

# Combine input selections together into a single dictionary mapping
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))

# Convert the dynamic dictionary directly into a 1-row DataFrame match for the pipeline
input_df = pd.DataFrame.from_dict(input_data, orient='index').T

# Ensure columns line up precisely with what the scikit-learn pipeline expects
expected_order = ['Education', 'City', 'Gender', 'EverBenched', 'Age', 'JoiningYear', 'ExperienceInCurrentDomain', 'PaymentTier']
input_df = input_df[expected_order]

# --- Prediction Action ---
if st.button('Predict'):
    prediction = predict_churn(input_df)[0]
    
    # Custom mapping matching your working dataset classification values (1 = Leave, 0 = Stay)
    translation_dict = {1: "Expected to Leave", 0: "Expected to Stay"}
    prediction_translate = translation_dict.get(int(prediction), "Unknown Status")
    
    # Styled output response
    if int(prediction) == 1:
        st.error(f'Prediction Output: **{prediction}** — The Employee is **{prediction_translate}**.')
    else:
        st.success(f'Prediction Output: **{prediction}** — The Employee is **{prediction_translate}**.')