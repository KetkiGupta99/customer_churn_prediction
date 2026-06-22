📉 Customer Churn Prediction: End-to-End Machine Learning & Business ROI

📌 Project Overview

This project tackles one of the most critical challenges in business: Customer Churn. Rather than just building a predictive model with high accuracy, this project bridges the gap between raw data, statistical analysis, machine learning trade-offs, and real-world financial impact (ROI).

The entire workflow is documented in a single Google Colab notebook, taking the raw Telco Customer Churn dataset and transforming it into a tiered business strategy.

🧠 Key Learnings & Project Philosophy

This project was built on three core data science principles:

Don't guess, prove it: We used statistical tests (T-Tests & Chi-Square) to mathematically prove which features actually cause churn before feeding them to a model.

Accuracy is a lie in imbalanced data: Predicting "Retained" 100% of the time yields high accuracy but destroys a business. We optimized for Recall (catching the churners) and managed the Precision-Recall Trade-off.

Good math can still lose money: A highly accurate model that flags too many "False Positives" will drain a marketing budget. Machine learning must be tied to a financial cost-benefit analysis.

🛠️ Step-by-Step Workflow & Methodology

Phase 1: Data Exploration & Wrangling

Before any analysis could begin, the raw data needed to be cleaned:

The TotalCharges Trap: The data loaded this column as an object (string) due to hidden blank spaces. I forced these into NaN values and converted the column to float64.

Target Variable Formatting: Mapped the target variable Churn from "Yes/No" to 1/0 for mathematical modeling.

Noise Reduction: Dropped the customerID column. Unique randomized strings offer zero predictive power and only confuse the algorithms.

Phase 2: Statistical Feature Selection

Instead of throwing all variables at a machine learning algorithm, I used scipy.stats to separate the signal from the noise ($\alpha = 0.05$):

Independent T-Tests (Continuous Variables): Tested tenure, MonthlyCharges, and TotalCharges. All proved highly statistically significant ($p < 0.05$).

Chi-Square Tests (Categorical Variables): Discovered that Contract (Month-to-Month) and InternetService (Fiber Optic) were massive drivers of churn.

The Drop: Found that gender and PhoneService were completely statistically insignificant. These were dropped from the dataset to prevent the model from learning random noise.

Phase 3: Data Preprocessing

One-Hot Encoding: Converted all remaining categorical variables into numeric binary columns (dummy variables) with drop_first=True to prevent multicollinearity.

Train-Test Split: Split the data 80/20. Crucial step: Used stratify=y to ensure the 73/27 split of Retained/Churned customers was perfectly preserved in both the training and testing sets.

Feature Scaling: Applied StandardScaler (Z-score normalization) exclusively to the continuous variables. Fit the scaler only on the training data to prevent data leakage.

Phase 4: Modeling & The Precision-Recall Trade-off

Three distinct modeling strategies were employed to understand the data:

Baseline Logistic Regression: * Used class_weight='balanced'.

Result: 79% Recall, 49% Precision. It caught almost all the churners but generated far too many false alarms (False Positives).

Random Forest (Threshold Tuning): * Extracted Feature Importances (Contract and tenure were the highest).

The Pivot: The default RF model was too conservative. I manually lowered the decision threshold from 0.50 to 0.30 ("If the model is even 30% sure they will churn, flag them").

Result: Hit the "Sweet Spot" with 71% Recall and 51% Precision.

XGBoost & SMOTE (Handling Imbalance):

Used SMOTE (Synthetic Minority Over-sampling Technique) to mathematically generate synthetic churners until the training data was a perfect 50/50 split.

Strict Protocol: SMOTE was applied only to the training dataset so the model was evaluated on 100% real, untouched human test data.

Trained an XGBoost Classifier on this balanced data to see if extreme gradient boosting could capture deeper patterns.

Phase 5: Business ROI & Financial Impact

The final and most important phase was translating the Random Forest Confusion Matrix into dollars.

The ROI Calculation: Assuming a $20 retention discount and a $65 average monthly revenue, I calculated the cost of False Positives (wasted discounts) vs. the revenue saved from True Positives.

The Discovery: A blanket $20 offer to all predicted churners resulted in a net loss. The cost of giving free money to False Positives outweighed the saved revenue.

The Solution (Tiered Interventions): Created a business rule strategy based on the model's probability scores:

Critical Risk (80%+ Probability): Offer the $20 discount.

High Risk (50-79% Probability): Offer a free temporary service upgrade (high perceived value, zero hard cost to the company).

Medium Risk (30-49% Probability): Send a personalized check-in email.

🚀 How to Run the Project

Clone this repository or download the Google Colab Notebook (.ipynb).

Download the Telco Customer Churn dataset (CSV).

Open the notebook in Google Colab or Jupyter Notebook.

Run the cells sequentially to watch the data transform from raw CSV into statistical proofs, machine learning models, and finally, a business ROI calculator.

💻 Tech Stack

Language: Python

Data Manipulation: pandas, numpy

Statistical Analysis: scipy.stats (T-Tests, Chi-Square)

Machine Learning: scikit-learn (Logistic Regression, Random Forest), xgboost

Imbalanced Data Handling: imblearn (SMOTE)

Visualizations: matplotlib, seaborn
