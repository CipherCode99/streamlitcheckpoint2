import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib 


@st.cache_data
def load_data():
    df = pd.read_csv("Financial_inclusion_dataset.csv")  
    return df

df = load_data()


st.write("### Dataset Preview")
st.write(df.head())


df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  


target_col = "bank_account"  
X = df.drop(columns=[target_col])
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "financial_model.pkl")

# Load the model
model = joblib.load("financial_model.pkl")


y_pred = model.predict(X_test)
st.write(f"### Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


st.write("## Predict Financial Inclusion")
input_features = {}

for col in X.columns:
    if col in label_encoders:  # If categorical, show dropdown
        options = label_encoders[col].classes_
        selected_option = st.selectbox(f"Select {col}", options)
        input_features[col] = label_encoders[col].transform([selected_option])[0]
    else: 
        input_features[col] = st.number_input(f"Enter {col}", value=float(df[col].median()))


input_df = pd.DataFrame([input_features])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Has a Bank Account" if prediction == 1 else "No Bank Account"
    st.write(f"### Prediction: {result}")
