# 📊 Customer Churn Analytics

A machine learning project for predicting customer churn in a telecommunications company.

---

## 🎯 Overview

Predicting which customers are likely to leave the company, using multiple ML models with hyperparameter tuning and ensemble techniques.

---

## 📁 Project Structure

```
├── ChurnCustomerAnalytics.ipynb    # Main notebook
├── Telco-Customer-Churn.csv        # Dataset
├── Design.py                        # Helper functions
└── README.md
```

---

## 📊 Dataset

**7,043 records** × **21 features**

- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, InternetService, StreamingTV, etc.
- **Account**: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: `Churn` (Yes/No)

---

## 🔧 Preprocessing

- Converted `TotalCharges` to numeric
- Merged `No phone/internet service` → `No`
- Dropped 11 missing values and `customerID`
- **Pipelines**: OneHotEncoder (nominal), OrdinalEncoder (ordinal), StandardScaler (numeric)

---

## 📈 Key EDA Findings

- **Month-to-month** contracts → highest churn
- **Fiber optic** users → higher churn probability
- **Low tenure** → higher churn risk
- **Electronic check** payment → higher churn

---

## 🤖 Models Trained

LogisticRegression · KNN · DecisionTree · RandomForest · MLP · GradientBoosting · AdaBoost · GaussianNB · **StackingClassifier**

Techniques: `train_test_split` · `5-Fold CV` · `RandomizedSearchCV` · `PCA`

---

## 🏆 Results

| Model | Test Accuracy | Test F1 |
|-------|---------------|---------|
| **StackingClassifier** ⭐ | **0.8053** | **0.597** |
| GradientBoosting | 0.7967 | 0.579 |
| AdaBoost | 0.7960 | 0.575 |
| LogisticRegression | 0.7953 | 0.568 |
| RandomForest | 0.7946 | 0.567 |

**Best Model: StackingClassifier** (LR + GradientBoosting + GaussianNB)

---

## ⚙️ Installation

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn
jupyter notebook ChurnCustomerAnalytics.ipynb
```

---

## 🛠 Technologies

Python 3.13 · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Jupyter


---
