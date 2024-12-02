# Heart Failure Prediction Using Machine Learning

## Overview
Cardiovascular diseases are the leading cause of mortality worldwide. This project leverages machine learning to predict the presence of heart disease based on patient data. By comparing six different models, we aim to identify the most effective approach for early diagnosis and intervention.

---

## Objectives
- Explore and preprocess the dataset to ensure data quality.
- Engineer features to enhance the performance of machine learning models.
- Implement and evaluate multiple machine learning algorithms for predicting heart disease.
- Provide insights and recommendations for practical applications in healthcare.

---

## Dataset
- **Source:** Kaggle  
- **Size:** 918 rows and 12 features  
- **Target Variable:** HeartDisease (binary: 0 = No, 1 = Yes)  
- Features include: Age, Cholesterol, MaxHR, Oldpeak, and more.

---

## Machine Learning Models
The following models were implemented and evaluated:
1. Logistic Regression
2. Naive Bayes
3. k-Nearest Neighbors (kNN)
4. Decision Tree
5. Random Forest
6. Neural Network

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Area Under the ROC Curve (AUC)

---

## Results
- **Best Overall Model:** Random Forest (AUC = 0.92)
- **Highest Precision:** Neural Network (Precision = 0.95)
- **Best Simplicity & Efficiency:** Naive Bayes (AUC = 0.94)

---

## Key Insights
- **Feature Importance:** Features like ST_Slope, MaxHR, and Oldpeak were the most predictive of heart disease.
- **Overfitting Challenges:** Neural Network exhibited slight overfitting, mitigated through regularization techniques.

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - Data Manipulation: Pandas, Numpy  
  - Visualization: Matplotlib, Seaborn  
  - Machine Learning: Scikit-learn, TensorFlow/Keras

---

## Repository Structure
```plaintext
📂 data
└── heart_disease.csv
├── Heart Failure Prediction.ipynb
├── 📂 reports
│   └── Project_Report.pdf
└── README.md
```
## How to Use
1. Clone the repository
```
git clone https://github.com/Trucodee/Heart-Failure-Prediction.git
```
2. Run the notebooks to reproduce the analysis and results

## Future Work
- Explore advanced models like XGBoost and LightGBM for improved performance.
- Address class imbalance using techniques like SMOTE.
- Validate the models on real-world patient data for clinical applicability.

## Acknowledgement
- Kaggle for providing the dataset.

