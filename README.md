# Vocal_Gender_Classification-ML
This project focuses on building a **binary classification model** to predict **gender (Male/Female)** based on vocal features. It's a great demonstration of how a Data Analyst can combine **EDA, feature selection, preprocessing, and model evaluation** to solve a real-world classification problem with high accuracy.

---

## 📌 Problem Statement

Classify the **gender of a speaker** using vocal frequency features. This type of solution has applications in speech recognition systems, conversational AI, and voice-based customer segmentation.

---

## 📊 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected shape, data types, missing values
- Explored feature distribution and class balance
- Visualized skewness and performed necessary transformations (log/sqrt)

### 2. Feature Engineering
- Removed highly correlated features (> 0.95)
- Applied statistical feature selection (RFE, SelectKBest)
- Scaled data using `StandardScaler` for better model performance

### 3. Model Building
Built and evaluated multiple models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

### 4. Model Evaluation
- Used metrics: Accuracy, Precision, Recall, F1-score
- Visualized **confusion matrices**
- Performed **cross-validation** to check model stability

### 5. Hyperparameter Tuning
- Used `GridSearchCV` to fine-tune SVM
- Achieved best score of **99.83%** with `C=1`, `gamma=0.01`, `kernel='rbf'`

---

## ✅ Key Highlights

- 🔍 **Feature Selection** and **Scaling** improved model efficiency
- 🧠 **SVM** was the best performing model with **>99% test accuracy**
- 📊 Demonstrated full ML pipeline suitable for production-level analysis

---

## 🧑‍💻 Tools & Technologies
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (`train_test_split`, `StandardScaler`, `GridSearchCV`, `classification_report`, etc.)
- Jupyter Notebook

---
