import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Sample dataset to simulate the model (you would replace this with your trained model)
data = pd.read_csv('https://raw.githubusercontent.com/TejeswarMudili/ML/main/insurance.csv')


# Define features (age, sex, bmi, children, smoker) and target variable (charges)
X = data[['age', 'sex', 'bmi', 'children', 'smoker']]
y = data["charges"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['sex', 'smoker'])],
    remainder='passthrough')  # Keep 'age', 'bmi', 'children' as numerical

# Gradient Boosting Regressor model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor(random_state=42))])

# Train the model
model.fit(X, y)

# Streamlit app
st.title("Insurance Charges Prediction")

# User input features
age = st.number_input('Age', min_value=18, max_value=100, value=25)
sex = st.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker (1 = Yes, 0 = No)', [1, 0])

# Convert numerical inputs back to categorical values expected by the model
sex = 'male' if sex == 1 else 'female'
smoker = 'yes' if smoker == 1 else 'no'

# When the user clicks the 'Predict' button
if st.button('Predict'):
    # Create input data as a DataFrame with column names
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker'])

    # Make prediction using the trained model
    prediction = model.predict(input_data)

    # Display the prediction
    st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")