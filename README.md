# 📉 Customer Churn Prediction: End-to-End Machine Learning & Business ROI

## 📌 Project Overview

This project tackles one of the most critical challenges in business: Customer Churn. Rather than just building a predictive model with high accuracy, this project bridges the gap between raw data, statistical analysis, machine learning trade-offs, and real-world financial impact (ROI).

The entire workflow is documented in a single Google Colab notebook, taking the raw Telco Customer Churn dataset and transforming it into a tiered business strategy.

---

## 🎯 Project Objectives

* Identify customers likely to churn.
* Validate important features using statistical testing.
* Compare multiple machine learning approaches.
* Handle class imbalance effectively.
* Optimize the Precision-Recall trade-off.
* Measure the financial impact of churn predictions.

---

## 🧠 Key Learnings & Project Philosophy

### 1. Don't Guess — Prove It

Statistical tests were used to determine whether features were genuinely associated with customer churn before model training.

### 2. Accuracy Can Be Misleading

Since churn datasets are imbalanced, accuracy alone is not a reliable metric. The project focuses on:

* Recall
* Precision
* F1 Score
* Business impact

### 3. Machine Learning Must Generate Business Value

A highly accurate model can still lose money if intervention costs exceed the revenue saved. Therefore, model predictions were evaluated through ROI analysis.

---

# 📊 Project Workflow

## Phase 1: Data Exploration & Cleaning

### Data Issues Addressed

#### TotalCharges Data Type Problem

* Column was loaded as `object` due to hidden blank spaces.
* Converted invalid entries to `NaN`.
* Cast column to `float64`.

#### Target Variable Transformation

* Converted:

  * `Yes → 1`
  * `No → 0`

#### Feature Reduction

Dropped `customerID` because:

* Unique for every customer
* No predictive value
* Adds unnecessary noise

---

## Phase 2: Statistical Feature Selection

To identify meaningful predictors, statistical hypothesis testing was performed at:

**Significance Level (α = 0.05)**

### Independent T-Tests (Numerical Features)

Tested:

* Tenure
* MonthlyCharges
* TotalCharges

### Findings

All three variables showed statistically significant differences between churned and retained customers.

**Result:** `p < 0.05`

---

### Chi-Square Tests (Categorical Features)

Analyzed relationships between categorical variables and churn.

### Strong Churn Drivers Identified

* Month-to-Month Contracts
* Fiber Optic Internet Service

### Statistically Insignificant Features

* Gender
* PhoneService

These features were removed to reduce noise and improve model performance.

---

## Phase 3: Data Preprocessing

### One-Hot Encoding

Converted categorical variables into numerical features using:

```python
pd.get_dummies(drop_first=True)
```

This prevents multicollinearity by avoiding the dummy variable trap.

### Train-Test Split

Dataset split:

* Training Set: 80%
* Testing Set: 20%

Used:

```python
stratify=y
```

to preserve the original churn distribution.

### Feature Scaling

Applied:

```python
StandardScaler()
```

on continuous variables.

Important:

* Scaler fitted only on training data.
* Prevented data leakage.

---

## Phase 4: Machine Learning Models

### 1️⃣ Logistic Regression (Baseline)

Used:

```python
class_weight='balanced'
```

### Performance

| Metric    | Score |
| --------- | ----- |
| Recall    | 79%   |
| Precision | 49%   |

### Interpretation

* Captured most churners.
* Generated many false positives.
* Expensive for retention campaigns.

---

### 2️⃣ Random Forest + Threshold Tuning

Random Forest initially produced conservative predictions.

### Feature Importance

Top predictors:

1. Contract Type
2. Tenure
3. Monthly Charges

### Threshold Optimization

Default threshold:

```python
0.50
```

Adjusted threshold:

```python
0.30
```

Business interpretation:

> "If the model is at least 30% confident a customer will churn, flag them for intervention."

### Performance

| Metric    | Score |
| --------- | ----- |
| Recall    | 71%   |
| Precision | 51%   |

### Outcome

Provided the best balance between:

* Capturing churners
* Controlling false positives
* Business usability

---

### 3️⃣ XGBoost + SMOTE

#### Handling Class Imbalance

Applied:

```python
SMOTE
```

to generate synthetic churn examples in the training dataset.

### Important Rule Followed

SMOTE was applied **only to training data**.

The test set remained:

* Real
* Unmodified
* Unbiased

### Benefits

* Balanced class distribution
* Improved learning of minority churn patterns
* Better generalization

---

## 📈 Model Evaluation

Evaluation metrics included:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Special emphasis was placed on:

### Recall

Because missing actual churners is costly.

### Precision

Because excessive false positives increase retention costs.

---

# 💰 Business ROI Analysis

The most valuable component of this project was translating model predictions into financial outcomes.

---

## Assumptions

### Customer Revenue

* Average Monthly Revenue = **$65**

### Retention Campaign Cost

* Discount Offered = **$20**

---

## Cost-Benefit Framework

### True Positive

Customer predicted to churn and successfully retained.

**Business Gain:**

* Revenue preserved

### False Positive

Customer predicted to churn but would have stayed anyway.

**Business Cost:**

* Unnecessary discount expense

---

# 🚀 Recommended Business Strategy

Instead of treating all predicted churners equally, use probability-based interventions.

| Risk Level       | Churn Probability | Action                         |
| ---------------- | ----------------- | ------------------------------ |
| 🔴 Critical Risk | 80%+              | Offer $20 retention discount   |
| 🟠 High Risk     | 50–79%            | Free temporary service upgrade |
| 🟡 Medium Risk   | 30–49%            | Personalized engagement email  |
| 🟢 Low Risk      | <30%              | No intervention                |

### Benefits

* Reduces unnecessary spending
* Improves retention efficiency
* Maximizes ROI
* Aligns machine learning with business goals

---

# 🚀 How to Run the Project

1. Clone this repository or download the Google Colab Notebook (.ipynb).
2. Download the Telco Customer Churn dataset (CSV).
3. Open the notebook in Google Colab or Jupyter Notebook.
4. Run the cells sequentially to watch the data transform from raw CSV into statistical proofs, machine learning models, and finally, a business ROI calculator.

---

💻 Tech Stack

Language: Python

Data Manipulation: pandas, numpy

Statistical Analysis: scipy.stats (T-Tests, Chi-Square)

Machine Learning: scikit-learn (Logistic Regression, Random Forest), xgboost

Imbalanced Data Handling: imblearn (SMOTE)

Visualizations: matplotlib, seaborn


