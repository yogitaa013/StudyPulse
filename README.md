# StudyPulse

StudyPulse is a simple end-to-end machine learning project that predicts a student’s academic performance category based on daily study habits, sleep, screen time, attendance, previous marks, and assignments completed. It combines data generation, model training, and a Streamlit web app so the workflow is easy to understand and present in interviews.

This project is a strong example of how machine learning can be applied in education to identify students who may need support early.

---

## Project Overview

StudyPulse predicts one of three classes:

- High
- Needs Improvement
- At Risk

The app takes student-related input values and returns a prediction along with simple suggestions for improvement.

---

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

---

## Project Structure

```text
StudyPulse/
├── app.py                  # Streamlit web application
├── generate_data.py        # Creates synthetic student dataset
├── train_model.py          # Trains the machine learning model
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained model
├── label_encoder.pkl       # Target label encoder
├── scaler.pkl              # Feature scaler
├── feature_importance.png  # Feature importance plot
└── data/
    └── student_data.csv    # Generated dataset
```

---

## How It Works

1. Data generation
   - The file generate_data.py creates a synthetic dataset of student records.
   - It generates values such as study hours, sleep hours, screen time, attendance, previous marks, and assignments completed.
   - A score is calculated and then grouped into High, Needs Improvement, or At Risk.

2. Model training
   - The file train_model.py loads the dataset.
   - It creates a new feature called distraction_score.
   - It scales the features and trains a Logistic Regression model.
   - It evaluates the model accuracy and saves the trained objects.

3. Web app prediction
   - The file app.py loads the trained model and provides a simple form in Streamlit.
   - Users enter student details and click Predict Performance.
   - The app returns the predicted category, confidence score, and improvement suggestions.

---

## Dataset Description

The dataset file is stored in data/student_data.csv and contains:

- study_hours
- sleep_hours
- screen_time
- attendance
- previous_marks
- assignments_completed
- result

The dataset is synthetic, which makes it ideal for learning and portfolio projects.

---

## Model Explanation

A Logistic Regression model is used because it is:

- easy to understand
- fast to train
- suitable for classification problems
- simple to explain in interviews

The project also uses StandardScaler to normalize feature values and LabelEncoder to convert text labels into numeric values for the model.

---

## How to Run Locally

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python generate_data.py
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Start the app

```bash
streamlit run app.py
```

---

## Interview-Ready Explanation

You can describe this project in an interview like this:

> I built a student performance prediction system using Python and Streamlit. The project starts by generating a synthetic dataset, trains a Logistic Regression model, and saves the trained artifacts for later use. I also built a web app so users can enter student details and receive predictions in real time.

### Strong talking points

- End-to-end machine learning workflow
- Data preprocessing and feature engineering
- Model training and evaluation
- Web app deployment using Streamlit
- Clear explanation of business use in education

---

## Possible Improvements

Future improvements could include:

- using a real dataset instead of synthetic data
- trying more advanced models such as Random Forest or XGBoost
- adding SHAP or explainability features
- deploying the app on Streamlit Cloud
- adding a dashboard for teachers and administrators

---

## Summary

StudyPulse is a beginner-friendly but interview-worthy machine learning project that demonstrates data generation, model training, evaluation, and deployment in a simple, practical way.
