import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import sklearn.compose._column_transformer
import sklearn.impute

# =========================================================================
# RUNTIME ENVIRONMENT FIXES (Version Mismatch & NumPy Compatibility)
# =========================================================================

# Fix 1: Mock missing internal class for ColumnTransformer cross-version alignment
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Fix 2: Dynamic patch to align SimpleImputer with newer NumPy type-casting standards (NEP 50)
original_transform = sklearn.impute.SimpleImputer.transform
def patched_transform(self, X, *args, **kwargs):
    if not hasattr(self, '_fill_dtype'):
        fallback_val = getattr(self, 'fill_value', 0)
        if fallback_val is None:
            fallback_val = 0
        self._fill_dtype = np.array(fallback_val).dtype
    return original_transform(self, X, *args, **kwargs)
sklearn.impute.SimpleImputer.transform = patched_transform


# =========================================================================
# 4. LOAD SCHEMA AND MODEL
# =========================================================================

# Load the pretrained pipeline using dill
with open('pipeline_class.pkl', 'rb') as file:
    model = dill.load(file)

# Load the feature schema dictionary using pickle
with open('class_feature_dict.pkl', 'rb') as file:
    my_feature_dict = pickle.load(file)

# Function to run data through the pipeline and return prediction
def predict_churn(data):
    prediction = model.predict(data)
    return prediction


# =========================================================================
# 5. HEADERS & IDENTIFICATION
# =========================================================================

st.title('Employee Churn Prediction App')
st.subheader('Created by Danish ul Khairi, CDA Student')
st.caption('Based on Employee Domain Dataset Assessment Framework')


# =========================================================================
# 6. USER INTERACTIONS (DYNAMIC UI INPUT ELEMENTS)
# =========================================================================

# --- Categorical Features Section ---
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}

# Extract structural columns and unique dropdown members from your schema
column_names = list(categorical_input.get('Column Name').values())
members_list = list(categorical_input.get('Members').values())

for i, col in enumerate(column_names):
    categorical_input_vals[col] = st.selectbox(col, members_list[i], key=col)

# --- Numerical Features Section ---
st.subheader('Numerical Features')
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}

for col in numerical_input.get('Column Name'):
    # Using integer stepping as Age, Years, and Tiers are discrete metrics
    numerical_input_vals[col] = st.number_input(col, step=1, value=0, key=col)


# =========================================================================
# 7. PROCESS INTERACTION VALUES FOR PREDICTION
# =========================================================================

# Combine user inputs into a unified dictionary format
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))

# Convert the interactive input dictionary into a single-row Pandas DataFrame
input_df = pd.DataFrame.from_dict(input_data, orient='index').T

# Explicit column alignment mapping to match what your scikit-learn pipeline expects
expected_order = ['Education', 'City', 'Gender', 'EverBenched', 'Age', 'JoiningYear', 'ExperienceInCurrentDomain', 'PaymentTier']
input_df = input_df[expected_order]


# =========================================================================
# 8. SHOW MODEL PREDICTION ON SUBMIT BUTTON
# =========================================================================

if st.button('Submit Assessment / Predict'):
    # Pass processed input data frame to the inference model
    prediction = predict_churn(input_df)[0]
    
    # Map raw target values (1 / 0) to descriptive status strings
    translation_dict = {1: "Expected to Leave (Attrition Risk)", 0: "Expected to Stay (Retention High)"}
    prediction_translate = translation_dict.get(int(prediction), "Unknown")
    
    # Render final prediction output visually on screen
    st.markdown("---")
    st.subheader("Model Assessment Output")
    
    if int(prediction) == 1:
        st.error(f'Prediction Label: **{prediction}** — The Employee is **{prediction_translate}**.')
    else:
        st.success(f'Prediction Label: **{prediction}** — The Employee is **{prediction_translate}**.')
