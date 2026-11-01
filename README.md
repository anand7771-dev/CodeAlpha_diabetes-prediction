
# 🩺 Disease Prediction System — ML-based Diabetes Risk Predictor

> **Python · Scikit-learn · Random Forest · Streamlit**

This project was developed during an internship at **Code Alpha** to predict the likelihood of **Diabetes** based on medical parameters using machine learning.

---

## ✅ Objective

Predict whether a patient is likely to have diabetes using structured clinical input features, by training and comparing multiple ML models and deploying the best-performing one in an interactive web application.

---

## 🔬 What Was Built

- **Built** a diabetes prediction model using the Pima Indians Diabetes Dataset, engineering features like glucose, BMI, insulin, and blood pressure to classify patient risk.
- **Trained and compared** Logistic Regression, Random Forest, and SVM models; selected **Random Forest as the final model** based on accuracy (**87%**).
- **Deployed** an interactive Streamlit app for real-time predictions, with saved model and scaler pipelines for consistent preprocessing.

---

## 📊 Features Used (Raw)

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (uU/ml) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Family history likelihood score |
| Age | Age in years |

## ⚗️ Engineered Features (+6)

| Feature | Formula | Clinical Meaning |
|---|---|---|
| Glucose_BMI | Glucose × BMI | Metabolic syndrome interaction |
| Glucose_Insulin | Glucose / (Insulin+1) | Insulin resistance proxy |
| High_Glucose | 1 if Glucose ≥ 126 | Diabetic glucose threshold |
| Obese | 1 if BMI ≥ 30 | Obesity flag |
| Senior | 1 if Age ≥ 45 | Age-related risk flag |
| BMI_Age | BMI × Age | Cumulative metabolic load |

---

## 🧠 Models Compared

| Model | Accuracy |
|---|---|
| Logistic Regression | ~79% |
| **Random Forest ✅ (Selected)** | **~87%** |
| SVM (RBF Kernel) | ~81% |

**Random Forest** was selected as the final model due to its highest accuracy on the test set.

---

## 📈 Final Model Performance (Random Forest)

- **Accuracy: ~87%**
- Trained on: Pima Indians Diabetes Dataset (768 patients)
- Validation: Stratified 5-fold cross validation
- Preprocessing: Class-stratified median imputation + StandardScaler
- Saved as: `best_model.pkl` + `scaler.pkl`

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional — model already saved)
```bash
python train_diabetes_model.py
```
This saves `best_model.pkl`, `scaler.pkl`, and `feature_names.pkl`.

### 3. Launch the Streamlit App
```bash
streamlit run diabetes_app.py
```

---

## 🗂️ Files

| File | Description |
|---|---|
| `train_diabetes_model.py` | Trains & compares LR, RF, SVM — saves best model |
| `diabetes_app.py` | Interactive Streamlit prediction app |
| `best_model.pkl` | Trained Random Forest classifier |
| `scaler.pkl` | Fitted StandardScaler for preprocessing |
| `feature_names.pkl` | Feature column names for inference |
| `requirements.txt` | Python dependencies |

---

## 🧪 Sample Input Ranges

| Feature | Normal Range | At-Risk |
|---|---|---|
| Glucose | 70–99 mg/dL (fasting) | 126+ |
| Blood Pressure | 60–80 mm Hg | 90+ |
| Skin Thickness | 10–25 mm | 35+ |
| Insulin | 16–166 uU/mL | 200+ |
| BMI | 18.5–24.9 | 30+ (obese) |
| Diabetes Pedigree | 0.0–0.5 | 0.8+ |
| Age | Any | 45+ (higher risk) |

---

🧑‍💻 Made with ❤️ during internship at **Code Alpha**.
