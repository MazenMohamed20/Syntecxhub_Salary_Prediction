# 💰 Salary Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.8991-brightgreen?style=for-the-badge)

> Predicting employee salaries using **Linear Regression** based on experience, education, gender, and job title.  
> Built as part of the **SyntecxHub Machine Learning Track — Week 1 Project**.

---

## 📌 Project Overview

This project builds a machine learning model that predicts the annual salary of employees based on key features like years of experience, education level, and job title.

| Metric | Value |
|--------|-------|
| **Model** | Linear Regression |
| **R² Score** | 0.8991 |
| **RMSE** | $15,551 |
| **Dataset** | Kaggle — Salary Prediction Dataset |
| **Training Samples** | 298 |
| **Test Samples** | 75 |

---

## 📂 Project Structure

```
Salary-Prediction/
│
├── main.py                          # Main training pipeline
├── predict.py                       # Load model & make predictions
├── Salary_Prediction_model.pkl      # Saved trained model
├── train.csv                        # Raw dataset
├── data_clear.csv                   # Cleaned dataset
└── README.md                        # You are here
```

---

## 🔄 Pipeline

```
Load Data → Clean & Handle Missing Values → Encode Categorical Features
    → Train/Test Split → Train Models → Compare → Save Best Model → Predict
```

### Step-by-step:

**1. Load & Explore**
```python
data = pd.read_csv('train.csv')
print(data.shape)         # (375, 6)
print(data.isnull().sum())
```

**2. Handle Missing Values**
```python
if data.isnull().sum().any():
    data_clear = data.dropna()   # 375 → 373 rows
```

**3. Encode Categorical Features**
```python
# Gender → LabelEncoder
data_clear['Gender'] = le.fit_transform(data_clear['Gender'])

# Education Level → Ordinal Encoding (order matters)
edu_order = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
data_clear['Education Level'] = data_clear['Education Level'].map(edu_order)

# Job Title → One-Hot Encoding
data_clear = pd.get_dummies(data_clear, columns=['Job Title'], drop_first=True)
```

**4. Train/Test Split**
```python
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42)
# Train: 298 | Test: 75
```

**5. Train & Compare 2 Models**
```python
# Model 1: Single Feature
model_single.fit(x_train[['Years of Experience']], y_train)

# Model 2: Multiple Features
model_multi.fit(x_train, y_train)
```

---

## 📊 Results

```
Single Feature (Years of Experience):
  RMSE : 15,551
  R²   : 0.8991

Multiple Features (All):
  RMSE : 18,858
  R²   : 0.8517
```

The **Single Feature model** outperformed the Multiple Features model, explaining **~90% of salary variance** using Years of Experience alone. The Multi-Feature model suffered due to high dimensionality from One-Hot Encoding Job Title (174 columns).

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Salary-Prediction.git
cd Salary-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn joblib
```

### 3. Train the model
```bash
python main.py
```

### 4. Make predictions
```bash
python predict.py
```

---

## 🧰 Tech Stack

- **Python 3.11**
- **Pandas** — Data loading & manipulation
- **NumPy** — Numerical operations
- **Scikit-Learn** — ML model & evaluation
- **Joblib** — Model serialization

---

## 📈 Key Insights

- `Years of Experience` is the strongest single predictor of salary
- **LabelEncoder** used for Gender & Ordinal Encoding for Education Level (order matters)
- One-Hot Encoding `Job Title` generated **174 columns**, causing slight overfitting in the multi-feature model
- Single Feature model achieved R² of **0.8991** — cleaner and more generalizable
- Only **2 rows** were dropped due to missing values (375 → 373)

---

## 🔮 Example Prediction

```python
# Employee with 5 years of experience
Predicted Salary: $65,634
```

---

## 👤 Author

**Mazen Mohamed**  
Machine Learning Trainee — SyntecxHub  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
