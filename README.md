# 💰 Salary Prediction — Machine Learning Project

A machine learning project that predicts employee salaries based on experience, education, gender, and job title using Linear Regression.

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Source | Kaggle — Salary Prediction Dataset |
| Raw Rows | 375 |
| After Cleaning | 373 |
| Features | 6 (Age, Gender, Education Level, Job Title, Years of Experience, Salary) |
| Target | Salary (USD) |

---

## ⚙️ Project Pipeline

```
Raw Data → Clean Data → Encode Features → Train/Test Split → Train Models → Evaluate → Save Best Model
```

### Steps:
1. **Load & Inspect** — Load CSV, check shape and data types
2. **Handle Missing Values** — Dropped 2 rows with null values (375 → 373)
3. **Encode Categorical Features:**
   - `Gender` → LabelEncoder (Male/Female)
   - `Education Level` → Ordinal Encoding (Bachelor's=0, Master's=1, PhD=2)
   - `Job Title` → One-Hot Encoding (→ 174 columns)
4. **Train/Test Split** — 80% Train (298) / 20% Test (75), random_state=42
5. **Train 2 Models** — Single Feature vs Multiple Features
6. **Evaluate** — RMSE & R² Score
7. **Save Best Model** — `.pkl` file using joblib

---

## 📈 Model Results

| Model | Features Used | RMSE | R² Score |
|-------|--------------|------|----------|
| Single Feature | Years of Experience only | $15,551 | 0.8991 |
| Multiple Features | All features | $18,858 | 0.8517 |

### 🏆 Best Model: **Single Feature (Years of Experience)**
> R² = **0.8991** — The model explains ~90% of salary variance using experience alone.

---

## 🔮 Example Prediction

```python
# Employee with 5 years of experience
Predicted Salary: $65,634
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue)

---

## 📁 Project Structure

```
Salary-Prediction/
│
├── train.csv                        # Raw dataset
├── data_clear.csv                   # Cleaned dataset
├── main.py                          # Training pipeline
├── predict.py                       # Load model & predict
├── Salary_Prediction_model.pkl      # Saved best model
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn joblib

# 2. Train the model
python main.py

# 3. Run predictions
python predict.py
```

---

## 👤 Author

**Mezo** — Trainee @ SYNTECXHUB  
*Building ML projects from scratch, step by step.*
