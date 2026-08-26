import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the company.")

# -----------------------------
# Create sample customer data
# -----------------------------

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "tenure": np.random.randint(1, 73, n),
    "monthly_charges": np.random.randint(20, 121, n),
    "total_charges": np.random.randint(100, 8000, n),
    "contract": np.random.choice(
        ["Month-to-month", "One year", "Two year"], n
    ),
    "internet_service": np.random.choice(
        ["DSL", "Fiber optic", "No"], n
    ),
    "tech_support": np.random.choice(
        ["Yes", "No"], n
    )
})

# Create churn pattern
data["churn"] = (
    (data["tenure"] < 12) &
    (data["monthly_charges"] > 70)
).astype(int)

# Add some random variation
random_indices = np.random.choice(n, 100, replace=False)
data.loc[random_indices, "churn"] = 1 - data.loc[random_indices, "churn"]

# -----------------------------
# Data preprocessing
# -----------------------------

df = pd.get_dummies(
    data,
    columns=["contract", "internet_service", "tech_support"],
    dtype=int
)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Machine Learning Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# User Input
# -----------------------------

st.header("👤 Enter Customer Details")

tenure = st.number_input(
    "Tenure (months)",
    min_value=1,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=20.0,
    max_value=120.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=100.0,
    max_value=8000.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Churn"):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "monthly_charges": [monthly_charges],
        "total_charges": [total_charges],
        "contract": [contract],
        "internet_service": [internet_service],
        "tech_support": [tech_support]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["contract", "internet_service", "tech_support"],
        dtype=int
    )

    input_data = input_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.write(
        f"Churn Probability: **{probability * 100:.2f}%**"
    )