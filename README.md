# Disease Risk Grouping and Diagnosis Prediction Using ML

An intelligent Machine Learning based healthcare prediction system that groups patients into risk profiles and predicts multiple diseases such as Heart Disease, Hypertension, and Diabetes Risk using clustering and classification techniques.

---

# Overview

This project combines both:

- Unsupervised Learning (K-Means Clustering)
- Supervised Learning (Random Forest Classification)

to analyze patient medical data, identify health-risk groups, and predict disease conditions with visualization dashboards.

The system also provides graphical insights such as:
- Cluster Analysis
- Confusion Matrix
- Feature Importance
- Model Accuracy Comparison

---

# Features

- Predicts:
  - Heart Disease
  - Hypertension
  - Diabetes Risk

- Uses K-Means Clustering for patient grouping
- Uses Random Forest for disease prediction
- Handles missing values automatically
- Generates a complete visualization dashboard
- Multi-output prediction system
- Simple clinical input interface

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |
| Seaborn | Statistical Graphs |
| Scikit-learn | Machine Learning |

---

# Machine Learning Concepts Used

## 1. K-Means Clustering
Groups patients into different risk profile clusters based on health similarities.

## 2. Random Forest Classifier
Predicts multiple disease conditions using ensemble learning.

## 3. StandardScaler
Normalizes medical data for better ML performance.

## 4. MultiOutputClassifier
Allows simultaneous prediction of multiple diseases.

---

# Dataset

Dataset used:
- `heart_disease_uci.csv`

The dataset contains medical attributes such as:
- Age
- Sex
- Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Fasting Blood Sugar

---

# Project Workflow

## Step 1: Data Loading
The dataset is loaded using Pandas.

## Step 2: Data Preprocessing
- Missing values handled using `SimpleImputer`
- Categorical data converted into numeric form
- Features cleaned and prepared

## Step 3: Feature Scaling
Data normalization performed using `StandardScaler`.

## Step 4: Patient Clustering
K-Means clustering divides patients into 3 health-risk groups.

## Step 5: Model Training
Random Forest model is trained for:
- Heart Disease Prediction
- Hypertension Prediction
- Diabetes Risk Prediction

## Step 6: User Input
The user enters clinical parameters manually.

## Step 7: Disease Prediction
The system predicts disease risks and displays diagnostic summaries.

## Step 8: Visualization Dashboard
The system generates:
- Cluster Plot
- Confusion Matrix
- Feature Importance Graph
- Accuracy Comparison Chart

---

# Input Parameters

| Parameter | Description |
|---|---|
| Age | Patient Age |
| Sex | Male/Female |
| Blood Pressure | BP Value |
| Cholesterol | Cholesterol Level |
| Max Heart Rate | Maximum Heart Rate |
| Fasting Blood Sugar | Diabetes Indicator |

---

# Example Input

```python
Age: 45
Sex: M
Blood Pressure: 140
Cholesterol: 220
Max Heart Rate: 160
Fasting Blood Sugar > 120? : Y
```

---

# Example Output

```text
DIAGNOSTIC SUMMARY
------------------
Risk Profile Group: 2

Heart Disease      : POSITIVE (Risk Detected)
Hypertension       : POSITIVE (Risk Detected)
Diabetes Risk      : Negative (Healthy)
```

---

# Visualization Dashboard

## 1. Patient Cluster Analysis
Shows where the patient belongs among different risk groups.

## 2. Confusion Matrix
Displays model reliability and prediction correctness.

## 3. Feature Importance
Shows which medical features influence predictions the most.

## 4. Accuracy Comparison
Compares prediction accuracy for all diseases.

---

# Advantages

- Multi-disease prediction system
- Visual explanation of AI predictions
- Easy to understand interface
- Combines clustering and classification
- Automatic preprocessing support

---

# Limitations

- Depends on dataset quality
- Not suitable for real clinical diagnosis
- Accuracy varies with training data
- Requires proper medical datasets

---

# Applications

- Healthcare Analytics
- Hospital Decision Support Systems
- Disease Risk Assessment
- Medical Research
- AI-based Clinical Prediction Systems

---

# Future Enhancements

- Deep Learning Integration
- Real-time IoT Health Monitoring
- Web-based Dashboard
- Mobile Application Support
- Larger Healthcare Datasets
- Advanced Explainable AI Techniques

---

# Conclusion

This project demonstrates how Machine Learning can assist healthcare systems by analyzing patient data, grouping patients into health-risk categories, and predicting multiple diseases simultaneously. The integration of clustering, classification, and visualization techniques makes the system informative, interactive, and useful for healthcare analytics and educational purposes.

---

# Author

Developed as a Machine Learning Healthcare Analytics Project using Python and Scikit-learn.
