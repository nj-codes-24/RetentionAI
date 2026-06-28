# 🔮 RetentionAI — Bank Customer Churn Prediction

> A production-style deep learning project that predicts whether a bank customer is likely to leave — built with an ANN in TensorFlow/Keras, a clean sklearn preprocessing pipeline, and an interactive Streamlit UI.

**🚀 Live Demo:** [https://retentify.streamlit.app/](https://retentify.streamlit.app/)

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [The Dataset](#the-dataset)
3. [The Core Problem: Why Preprocessing Matters](#the-core-problem-why-preprocessing-matters)
4. [Preprocessing Pipeline — The Right Way](#preprocessing-pipeline--the-right-way)
5. [Building the ANN](#building-the-ann)
6. [Training Strategies: Early Stopping & TensorBoard](#training-strategies-early-stopping--tensorboard)
7. [Saving & Loading Artifacts](#saving--loading-artifacts)
8. [Making Predictions](#making-predictions)
9. [The Streamlit App](#the-streamlit-app)
10. [Project File Structure](#project-file-structure)
11. [How to Run This Locally](#how-to-run-this-locally)
12. [Key Concepts Cheat Sheet](#key-concepts-cheat-sheet)

---

## What This Project Does

Banks lose money when customers close their accounts — this is called **churn**. Identifying which customers are *likely* to churn before they actually do allows a bank to intervene (offer better rates, personalised outreach, etc.).

This project trains an **Artificial Neural Network (ANN)** on historical customer data to output a single number: the **probability that a given customer will churn**. If that probability is above 0.5, we flag them as at risk.

The full pipeline is:

```
Raw CSV Data
     ↓
Feature Engineering (drop irrelevant columns)
     ↓
sklearn ColumnTransformer (impute → encode → scale)
     ↓
Train/Test Split
     ↓
ANN (TensorFlow/Keras) — 3 layer network
     ↓
Saved model.h5 + preprocessor.pkl
     ↓
Streamlit Web App (user inputs → prediction)
```

---

## The Dataset

**File:** `Churn_Modelling.csv`

The dataset contains 10,000 rows of fictional bank customer records. Here is what each column means:

| Column | Type | What It Represents |
|---|---|---|
| `RowNumber` | int | Just an index — **irrelevant**, dropped |
| `CustomerId` | int | A unique ID — **irrelevant**, dropped |
| `Surname` | string | Customer's last name — **irrelevant**, dropped |
| `CreditScore` | int | How creditworthy the customer is (300–900) |
| `Geography` | string | Country: France, Germany, or Spain |
| `Gender` | string | Male or Female |
| `Age` | int | Customer's age |
| `Tenure` | int | How many years they've been a customer (0–10) |
| `Balance` | float | Account balance in dollars |
| `NumOfProducts` | int | How many bank products they use (1–4) |
| `HasCrCard` | int | Whether they have a credit card (0 or 1) |
| `IsActiveMember` | int | Whether they've been active recently (0 or 1) |
| `EstimatedSalary` | float | Annual salary estimate |
| `Exited` | int | **TARGET**: 1 = churned, 0 = stayed |

The first three columns (`RowNumber`, `CustomerId`, `Surname`) carry zero predictive signal — they're identifiers, not features. We drop them immediately.

---

## The Core Problem: Why Preprocessing Matters

Neural networks (and most ML models) can't handle raw data as-is. There are two main issues:

### Problem 1 — Categorical Columns

The model can't do math on strings like `"France"` or `"Male"`. These need to be converted to numbers.

There are two ways to encode categorical data:

**Label Encoding** — assigns each category an integer: `France=0, Germany=1, Spain=2`. This is fine for binary categories (`Male=0, Female=1`) but introduces a false ordering for multi-class columns. Why would the model treat Germany as numerically "between" France and Spain?

**One-Hot Encoding** — creates a separate binary column for each category. `Geography` becomes three columns: `Geography_France`, `Geography_Germany`, `Geography_Spain`. Each row gets a 1 in the relevant column and 0 elsewhere. This removes the false ordering problem.

In this project:
- `Gender` → **One-Hot Encoded** with `drop="if_binary"` (which drops one column when there are only 2 categories, avoiding multicollinearity — if `Gender_Male=1`, we already know `Gender_Female=0`)
- `Geography` → **One-Hot Encoded**, producing 3 columns

### Problem 2 — Feature Scale

`Balance` can be in the hundreds of thousands. `HasCrCard` is 0 or 1. If we feed these raw into a neural network, the large-scale features will dominate gradient updates and the model learns very slowly or not at all.

**StandardScaler** solves this by transforming each feature to have **mean = 0** and **standard deviation = 1**:

```
z = (x - mean) / std
```

After scaling, all numeric features live on a comparable scale, making gradient descent converge much faster.

---

## Preprocessing Pipeline — The Right Way

Rather than doing preprocessing steps manually (fit on train, transform on test separately, risk leakage), we use sklearn's `Pipeline` and `ColumnTransformer` to bundle everything into a single, reusable object.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_col = ['CreditScore', 'Age', 'Tenure', 'Balance',
               'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
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
```

Let's break down exactly what is happening here:

**`SimpleImputer`** — handles missing values. For numeric columns we fill gaps with the column mean. For categorical columns we fill with the most frequent value. This dataset doesn't have missing values, but including imputers is professional practice — real-world data almost always does.

**`Pipeline(steps=[...])`** — chains transformations in sequence. For numeric columns: impute first, then scale. For categorical columns: impute first, then encode. Each step's output feeds directly into the next step.

**`ColumnTransformer(transformers=[...])`** — applies *different* pipelines to *different* subsets of columns, then horizontally concatenates the results. This is the core of clean preprocessing. You tell it: "apply `numeric_transform` to these columns, apply `categorical_transform` to those columns."

**`remainder="passthrough"`** — any column not explicitly listed in the transformers passes through unchanged. Safe default.

### The Critical Rule: Fit on Train, Transform on Both

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)  # learns stats from train set
X_test_processed  = preprocessor.transform(X_test)        # uses train set stats on test set
```

**Why this order matters:** If you fit the scaler on the full dataset before splitting, the test set's statistics "leak" into the scaler's learned mean/std. During evaluation, the model has indirectly seen information from the test set. This inflates your metrics and gives a false sense of performance. Always `fit_transform` on train only, `transform` on test.

---

## Building the ANN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Architecture Walkthrough

**Input Layer** — `X_train_processed.shape[1]` gives the number of features after encoding. With 8 numeric columns + 3 one-hot columns from Geography + 1 binary column from Gender = **12 input features**.

**Hidden Layer 1 — 64 neurons, ReLU activation**

ReLU (Rectified Linear Unit) is defined as `f(x) = max(0, x)`. It's the most commonly used activation because:
- It's computationally cheap (just a max operation)
- It doesn't suffer from the vanishing gradient problem like sigmoid/tanh in deep networks
- It introduces non-linearity, which lets the network learn complex patterns

**Hidden Layer 2 — 32 neurons, ReLU activation**

A smaller second layer creates a funnel-like structure. The network learns high-level patterns in the first layer, then distills them further in the second. The narrowing also acts as mild regularisation.

**Output Layer — 1 neuron, Sigmoid activation**

Sigmoid squashes any input to a value between 0 and 1: `f(x) = 1 / (1 + e^(-x))`. This is exactly what we want — a probability. We interpret it as "the probability this customer churns."

### Compilation

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=['accuracy']
)
```

**Adam Optimizer** — Adaptive Moment Estimation. It adapts the learning rate per parameter using estimates of first and second moments of gradients. In practice, Adam converges faster and more reliably than vanilla SGD for most problems. `learning_rate=0.01` is the step size during gradient descent.

**Binary Cross-Entropy Loss** — the standard loss function for binary classification. For each sample it computes:

```
loss = - [y * log(p) + (1 - y) * log(1 - p)]
```

where `y` is the true label (0 or 1) and `p` is the predicted probability. When the model is confident and correct, loss is near 0. When it's confident and wrong, loss explodes — this creates a strong gradient signal to correct the error.

**Why not MSE for classification?** Mean Squared Error penalises all errors equally regardless of confidence. Cross-entropy penalises confident wrong predictions much more severely, which is the right incentive for a classifier.

---

## Training Strategies: Early Stopping & TensorBoard

### Early Stopping

```python
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

The problem with training for a fixed number of epochs: the model keeps updating weights even after validation performance has stopped improving (or started degrading). This is **overfitting** — the model memorises the training data and performs worse on unseen data.

Early stopping monitors `val_loss` (loss on the test/validation set). If it doesn't improve for `patience=10` consecutive epochs, training stops. `restore_best_weights=True` rewinds the model to the weights from the epoch with the best `val_loss`, not the final epoch — this is important because the final epoch is often slightly overfit compared to the best epoch.

### TensorBoard (Optional)

```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
```

TensorBoard is TensorFlow's visualisation toolkit. It logs training metrics (loss, accuracy) at each epoch to a directory. You can then launch a local web interface to see learning curves in real time:

```bash
tensorboard --logdir logs/fit
```

`histogram_freq=1` additionally logs weight and bias distributions per layer at every epoch, which helps debug things like vanishing gradients or dead neurons.

### Training

```python
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_test_processed, y_test),
    epochs=100,
    callbacks=[early_stopping_callback]
)
```

`validation_data` tells Keras to evaluate the model on the test set at the end of each epoch without using that data for gradient updates. This gives us the `val_loss` signal that Early Stopping monitors.

---

## Saving & Loading Artifacts

After training, we save two things:

```python
# Save the trained neural network
model.save('model.h5')

# Save the entire preprocessing pipeline
with open('preprocessor.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)
```

**Why save the preprocessor?** When you deploy and get a new customer's raw data, you need to apply *exactly the same transformations* with *exactly the same statistics* (the same mean, std, encoder categories) that were learned during training. If you re-fit the preprocessor on new data, the scaling will be different and the model will receive inputs from a different distribution than it was trained on — predictions will be garbage.

The `preprocessor.pkl` file is the memory of what the training data looked like. It must travel alongside the model.

**`.h5` format** — HDF5 is a hierarchical data format that stores the model architecture, weights, and training configuration in a single file. Loading it is one line:

```python
model = tf.keras.models.load_model('model.h5')
```

---

## Making Predictions

```python
import pickle
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Raw input — exactly as you'd receive from a user form
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

input_df = pd.DataFrame([input_data])

# The preprocessor handles encoding + scaling in one step
input_scaled = preprocessor.transform(input_df)

prediction = model.predict(input_scaled)
churn_proba = prediction[0][0]

print(f"Churn Probability: {churn_proba:.4f}")
if churn_proba > 0.5:
    print("High risk of churning.")
else:
    print("Likely to be retained.")
```

Notice that inference is clean. We never manually encode or scale — the pipeline does it. This is the entire payoff of using `ColumnTransformer`: at inference time, you just call `.transform()` and everything is handled consistently.

---

## The Streamlit App

`app.py` wraps the prediction logic in a web UI. The user fills out a form with customer details, clicks a button, and sees the churn probability and a risk assessment.

Key design decisions:
- `@st.cache_resource` loads the model and preprocessor once at startup and caches them — not on every user interaction
- Inputs are collected through sliders, dropdowns, and number inputs that match the exact data types the model expects
- The raw form inputs are assembled into a `pd.DataFrame` with column names that match the training data exactly — the `ColumnTransformer` identifies columns by name
- Risk level is shown as a colour-coded result with a probability gauge

---

## Project File Structure

```
RetentionAI/
├── Churn_Modelling.csv     # Raw dataset (10,000 customer records)
├── exploration.py          # Data loading, preprocessing pipeline, ANN training
├── prediction.py           # Standalone script for making a single prediction
├── app.py                  # Streamlit web application
├── model.h5                # Saved trained ANN weights + architecture
└── preprocessor.pkl        # Saved sklearn ColumnTransformer pipeline
```

---

## How to Run This Locally

**1. Install dependencies**

```bash
pip install tensorflow scikit-learn pandas numpy streamlit
```

**2. Train the model (or skip if model.h5 already exists)**

```bash
python exploration.py
```

This reads `Churn_Modelling.csv`, preprocesses the data, trains the ANN, and saves `model.h5` and `preprocessor.pkl`.

**3. Test a prediction from the command line**

```bash
python prediction.py
```

**4. Launch the web app**

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Key Concepts Cheat Sheet

| Concept | One-line explanation |
|---|---|
| **Churn** | A customer stopping their relationship with the bank |
| **ANN** | A series of connected layers of neurons that learn patterns from data |
| **ReLU** | Activation function: output = max(0, input). Introduces non-linearity |
| **Sigmoid** | Squashes any number to (0, 1). Used in output layer for probability |
| **Binary Cross-Entropy** | Loss function for binary classification. Penalises confident wrong predictions hard |
| **Adam** | Adaptive optimiser that adjusts learning rates per parameter |
| **StandardScaler** | Transforms features to mean=0, std=1 so no feature dominates |
| **One-Hot Encoding** | Converts a categorical column into N binary columns (one per category) |
| **ColumnTransformer** | Applies different transformations to different subsets of columns |
| **Pipeline** | Chains multiple transformations so fit/transform happen in sequence |
| **Early Stopping** | Stops training when validation loss stops improving to prevent overfitting |
| **Data Leakage** | When test set information contaminates training — always fit preprocessor on train only |
| **`.h5`** | HDF5 format used to save a Keras model's architecture + weights |
| **`pickle`** | Python serialisation used to save the preprocessor object to disk |
| **`val_loss`** | Loss computed on the validation/test set — the true measure of generalisation |
