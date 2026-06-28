import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

## Streamlit app
st.title('Retention AI')

# User input
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create a button to trigger the prediction
if st.button('Predict Retention'):
    
    # Prepare the input data (Must match exact columns used during training)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Transform the input data using our single preprocessor pipeline
    input_data_scaled = preprocessor.transform(input_data)

    # Predict retention
    prediction = model.predict(input_data_scaled)
    churn_proba = prediction[0][0]
    retention_proba = 1 - churn_proba

    st.write(f'Retention Probability: {retention_proba:.2f}')

    # Visual feedback based on the result
    if retention_proba < 0.5:
        st.error('The customer is at high risk of leaving.')
    else:
        st.success('The customer is likely to be retained.')