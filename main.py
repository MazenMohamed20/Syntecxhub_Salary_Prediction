import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, metrics, preprocessing
import joblib

data = pd.read_csv(r'E:\SYNTECXHUB\Task_1_Salary Prediction\train.csv')

if data.isnull().sum().any():
    data_clear = data.dropna()
    data_clear.to_csv(r'E:\SYNTECXHUB\Task_1_Salary Prediction\data_clear.csv', index=False)

le = preprocessing.LabelEncoder()
data_clear = data_clear.copy()
data_clear['Gender'] = le.fit_transform(data_clear['Gender'])

edu_order = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
data_clear['Education Level'] = data_clear['Education Level'].map(edu_order)
data_clear = pd.get_dummies(data_clear, columns=['Job Title'], drop_first=True)

x = data_clear.drop('Salary', axis=1)
y = data_clear['Salary']

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42
)

# =====================
# Model 1: Single Feature
# =====================
model_single = linear_model.LinearRegression()
model_single.fit(x_train[['Years of Experience']], y_train)
y_pred_single = model_single.predict(x_test[['Years of Experience']])

rmse_single = np.sqrt(metrics.mean_squared_error(y_test, y_pred_single))
r2_single = metrics.r2_score(y_test, y_pred_single)

print("Single Feature (Years of Experience):")
print(f"  RMSE : {rmse_single:,.0f}")
print(f"  R2   : {r2_single:.4f}")

# =====================
# Model 2: Multiple Features
# =====================
model_multi = linear_model.LinearRegression()
model_multi.fit(x_train, y_train)
y_pred_multi = model_multi.predict(x_test)

rmse_multi = np.sqrt(metrics.mean_squared_error(y_test, y_pred_multi))
r2_multi = metrics.r2_score(y_test, y_pred_multi)

print("\nMultiple Features (All):")
print(f"  RMSE : {rmse_multi:,.0f}")
print(f"  R2   : {r2_multi:.4f}")

# =====================
# Comparison and Saving Best Model
# =====================
print("\n--- Comparison ---")
print(f"Single  -> RMSE: {rmse_single:,.0f} | R2: {r2_single:.4f}")
print(f"Multiple -> RMSE: {rmse_multi:,.0f} | R2: {r2_multi:.4f}")


if r2_multi > r2_single:
    joblib.dump(model_multi, r'E:\SYNTECXHUB\Task_1_Salary Prediction\Salary_Prediction_model.pkl')
    print("\nBest Model: Multiple Features ✅ (saved)")
else:
    joblib.dump(model_single, r'E:\SYNTECXHUB\Task_1_Salary Prediction\Salary_Prediction_model.pkl')
    print("\nBest Model: Single Feature ✅ (saved)")