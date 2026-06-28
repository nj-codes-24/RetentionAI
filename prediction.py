import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# 1. Load the trained model and the unified preprocessor
model = load_model('model.h5')

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# 2. Example input data (Raw format, exactly as it comes from a user/app)
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

# 3. Convert the dictionary into a Pandas DataFrame
input_df = pd.DataFrame([input_data])

# 4. Process the data
# The preprocessor automatically handles the one-hot encoding, label encoding, and scaling!
input_scaled = preprocessor.transform(input_df)

# 5. Predict churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# 6. Output the results
print(f"Churn Probability: {prediction_proba:.4f}")

if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')