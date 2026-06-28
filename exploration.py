import pandas as pd
import numpy as np
import pickle
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

X = data.drop('Exited', axis=1)
y = data['Exited']

numeric_col = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_col = ['Geography', 'Gender']

numeric_transform = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler())
])

categorical_transform = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(drop="if_binary", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transform, numeric_col),
    ("categorical", categorical_transform, categorical_col)
], remainder="passthrough")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

with open('preprocessor.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)

model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=['accuracy']
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_test_processed, y_test),
    epochs=100,
    callbacks=[early_stopping_callback] 
)

model.save('model.h5')