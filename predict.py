import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib

# Load the best model
model = joblib.load(r'E:\SYNTECXHUB\Task_1_Salary Prediction\Salary_Prediction_model.pkl')

# example input for prediction
example = pd.DataFrame({
    'Years of Experience': [5]
})

predicted_salary = model.predict(example)
print(f"Predicted Salary: ${predicted_salary[0]:,.0f}")