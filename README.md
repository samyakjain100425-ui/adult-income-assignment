# 💼 Adult Income Classification Using Multiple Machine Learning Models

## 📌 Problem Statement

The objective of this project is to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes. 

This is formulated as a supervised binary classification problem where the target variable `income` has two classes:

- <=50K
- >50K

---

## 📊 Dataset Description

The dataset used in this project is the Adult Census Income dataset sourced from the Kaggle.

### Dataset Characteristics:
- Type: Binary Classification
- Total Instances: ~48,842
- Total Features: 14
- Target Variable: income
- Feature Types: Categorical and Numerical

### Feature List:

- age  
- workclass  
- fnlwgt  
- education  
- educational-num  
- marital-status  
- occupation  
- relationship  
- race  
- gender  
- capital-gain  
- capital-loss  
- hours-per-week  
- native-country  
- income (Target Variable)

### Data Preprocessing Steps:
- Removed missing values
- Trimmed extra spaces
- Applied one-hot encoding to categorical variables
- Standardized numerical features where required
- Split dataset into 80% training and 20% testing

---

## 🤖 Machine Learning Models Implemented

The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## 📈 Evaluation Metrics Used

Each model was evaluated using the following performance metrics:

- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 📊 Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|----------|------|----------|--------|------|------|
| Logistic Regression | 0.8537 | 0.9055 | 0.7411 | 0.5975 | 0.6616 | 0.5750 |
| Decision Tree | 0.8197 | 0.7569 | 0.6203 | 0.6364 | 0.6282 | 0.5094 |
| KNN | 0.8263 | 0.8379 | 0.6561 | 0.5761 | 0.6135 | 0.5039 |
| Naive Bayes | 0.5675 | 0.7971 | 0.3505 | 0.9461 | 0.5115 | 0.3520 |
| Random Forest | 0.8639 | 0.9168 | 0.7975 | 0.5778 | 0.6701 | 0.5992 |
| XGBoost | 0.8759 | 0.9297 | 0.7920 | 0.6531 | 0.7159 | 0.6423 |

---

## 🔍 Model Performance Observations

| ML Model | Observation |
|------------|------------|
| Logistic Regression | Provides strong baseline performance with high AUC but slightly lower recall for the high-income class. |
| Decision Tree | Moderate performance but prone to overfitting and less stable than ensemble methods. |
| KNN | Sensitive to scaling and hyperparameters; performs reasonably but not optimal. |
| Naive Bayes | Shows high recall but low precision and accuracy due to independence assumption. |
| Random Forest | Better generalization through ensemble averaging and improved balance between precision and recall. |
| XGBoost | Best overall performer with highest Accuracy, AUC, F1, and MCC. Effectively captures complex patterns in data. |

### Key Insight

Ensemble models (Random Forest and XGBoost) outperform individual classifiers due to improved generalization and reduced variance. XGBoost achieved the best overall performance on this dataset.

---

## 🌐 Streamlit Web Application

An interactive Streamlit web application was developed and deployed using Streamlit Community Cloud.

### Application Features:
- Upload test dataset (CSV format)
- Select trained model from dropdown
- Display evaluation metrics
- Show confusion matrix
- Show classification report
- Preview predictions

---

## 📁 Project Structure

project-folder/
│
├── data/
│   └── adult.csv
│
├── model/
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── app.py
├── train_models.py
├── requirements.txt
└── README.md  

---

## 🎓 Conclusion

This project demonstrates implementation and comparison of multiple machine learning classification models on a structured tabular dataset. 

Among all models, XGBoost achieved the best overall performance, followed by Random Forest. The results highlight the effectiveness of ensemble learning techniques in handling complex classification tasks.
