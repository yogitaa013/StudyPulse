# StudyPulse

StudyPulse is a beginner-friendly machine learning project that predicts a student’s academic performance category based on study habits, sleep, screen time, attendance, previous marks, and assignments completed. The project includes synthetic data generation, model training, and a Streamlit-based web application so users can interact with the model easily.

This project is useful for learning the full machine learning workflow in a simple and practical way. It shows problem understanding, data preparation, model building, evaluation, and deployment.

---

## 1. Project Goal

The main goal of StudyPulse is to classify students into one of three performance categories:

- High
- Needs Improvement
- At Risk

The system helps identify students who may need extra support earlier, based on their daily habits and academic indicators.

---

## 2. Why This Project Matters

Education data can be used to understand student behavior and detect patterns that may affect academic performance. This project demonstrates how machine learning can support educators and learners by providing a simple prediction model that can guide decisions.

In real life, this idea could be extended into:

- student performance monitoring systems
- academic warning dashboards
- personalized learning recommendations
- early intervention tools for schools

---

## 3. Project Overview

StudyPulse is an end-to-end mini machine learning application with three main parts:

1. Data generation
2. Model training
3. Web-based prediction app

It is designed to be simple enough for beginners while still being useful for understanding a complete machine learning workflow.

### Workflow

1. The user provides student-related inputs through the Streamlit app.
2. The app prepares the input values and creates a distraction score.
3. The trained model predicts whether the student is High, Needs Improvement, or At Risk.
4. The app displays the prediction and a few improvement suggestions.

---

## 4. Technologies Used

The project uses:

- Python for development
- Streamlit for the interactive web interface
- pandas for data handling
- numpy for numerical operations
- scikit-learn for building the machine learning model
- joblib for saving and loading the trained model
- matplotlib for visualizing feature importance

---

## 5. Project Structure

```text
StudyPulse/
├── app.py                  # Streamlit web application
├── generate_data.py        # Generates synthetic student data
├── train_model.py          # Trains the model and saves artifacts
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained machine learning model
├── label_encoder.pkl       # Encodes target labels
├── scaler.pkl              # Standardizes input features
├── feature_importance.png  # Feature importance visualization
└── data/
    └── student_data.csv    # Dataset used by the model
```

---

## 6. File Explanation

### app.py
This is the main Streamlit app. It loads the trained model and allows users to enter student details. After clicking the prediction button, the app displays:

- predicted performance category
- confidence score
- simple improvement suggestions

### generate_data.py
This script creates a synthetic dataset of student records. It generates random values for behavior and academic factors and assigns a performance label based on a custom scoring formula.

### train_model.py
This script:

- loads the CSV dataset
- creates a new feature named distraction_score
- splits the data into training and testing sets
- trains a Logistic Regression classifier
- evaluates accuracy
- saves the trained model and preprocessing files

### requirements.txt
This file lists all the Python packages required to run the project.

---

## 7. Dataset Details

The dataset is stored in data/student_data.csv and contains the following columns:

- study_hours
- sleep_hours
- screen_time
- attendance
- previous_marks
- assignments_completed
- result

### What the columns mean

- study_hours: average hours a student studies daily
- sleep_hours: average hours of sleep per day
- screen_time: daily screen time in hours
- attendance: percentage of attendance
- previous_marks: previous academic marks percentage
- assignments_completed: number of assignments completed out of 10
- result: target label showing the student’s performance category

### Note
The dataset used in this project is synthetic, which means it is created for demonstration purposes and learning. In a real-world project, you would replace it with real student data.

---

## 8. Machine Learning Approach

The project uses a Logistic Regression classifier because it is simple, efficient, and easy to explain.

### Why this model was chosen

- It is suitable for classification problems.
- It trains quickly.
- It is easy to understand during interviews.
- It works well for a beginner project.

### Preprocessing steps

Before training, the project performs the following:

- feature scaling using StandardScaler
- label encoding using LabelEncoder
- feature engineering by creating distraction_score

The distraction_score is calculated as:

```python
screen_time / study_hours
```

This helps capture the relationship between excessive screen time and academic effort.

---

## 9. How the Project Runs

### Step 1: Generate data
Run:

```bash
python generate_data.py
```

This creates the student dataset in the data folder.

### Step 2: Train the model
Run:

```bash
python train_model.py
```

This trains the classifier and saves the model files.

### Step 3: Launch the app
Run:

```bash
streamlit run app.py
```

The app opens in the browser, where the user can enter student details and get a prediction.

---

## 10. How to Setup the Project Locally

### Prerequisites
Make sure Python is installed on your system.

### Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python generate_data.py
python train_model.py
streamlit run app.py
```

---

## 11. What the User Sees in the App

When the app runs, the user sees a form with fields such as:

- study hours
- sleep hours
- screen time
- attendance percentage
- previous marks
- assignments completed

After submitting the values, the app predicts one of the three categories and shows a confidence score.

It may also provide suggestions such as:

- increase study time
- reduce screen time
- improve attendance
- maintain consistent sleep

---

## 12. Expected Output

The app will return outputs like:

- High
- Needs Improvement
- At Risk

Depending on the user input, the model may show a confidence percentage and improvement suggestions.

---

## 13. Business Value

This project demonstrates how data science can be used in education. In a real-world setting, the same idea could help:

- teachers identify at-risk students earlier
- schools monitor performance trends
- educational apps offer personalized support
- parents better understand academic challenges

---


## 14. Possible Improvements

This project can be extended in many ways:

- use a real student dataset
- try better models like Random Forest or XGBoost
- add explainability using SHAP
- deploy the app on Streamlit Cloud
- add a dashboard for schools or teachers
- save predictions in a database for analysis

---

## 15. Summary

StudyPulse is a simple, practical, and interview-friendly project that shows how a machine learning model can be built, trained, and used through a web application. It is ideal for beginners who want to understand the complete workflow of a real AI project.
