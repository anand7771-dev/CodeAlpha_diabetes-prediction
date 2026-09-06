
# 🩺 Disease Prediction using Machine Learning

This project is developed as part of an internship to predict the likelihood of **Diabetes** based on medical parameters using machine learning.

## ✅ Objective
Predict whether a patient is likely to have diabetes using structured input features from medical data.

## 📊 Features Used
- **Pregnancies**
- **Glucose**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin**
- **BMI (Body Mass Index)**
- **Diabetes Pedigree Function**
- **Age**

## 🧠 Models Used
- Logistic Regression
- Random Forest (final model)
- XGBoost (optional)
- Support Vector Machine (optional)

Model was trained using the **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository.

## 📈 Model Performance
- Accuracy: ~73% (Random Forest)

## 🚀 How to Run

### 1. Train the Model
```
python train_diabetes_model.py
```
This will save `best_model.pkl` and `scaler.pkl`.

### 2. Launch Streamlit App
```
streamlit run diabetes_app.py
```

## 🗂️ Files Included
- `train_diabetes_model.py` – Model training script
- `diabetes_app.py` – Streamlit frontend
- `best_model.pkl` – Trained ML model
- `scaler.pkl` – Scaler for preprocessing input
- `requirements.txt` – Python dependencies

## 🧪 Sample Input Ranges

| Feature                  | Normal Range             |
|--------------------------|--------------------------|
| Glucose                  | 70–140 mg/dL             |
| Blood Pressure           | 80–120 mmHg              |
| Skin Thickness           | Around 20 mm             |
| Insulin                  | 16–166 μU/mL             |
| BMI                      | 18.5–24.9                |
| Diabetes Pedigree Func.  | 0.2–1.0                  |

---

🧑‍💻 Made with ❤️ during internship at Code Alpha.
