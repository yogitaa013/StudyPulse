# StudyPulse - Student Performance Predictor

StudyPulse is a **Machine Learning web application** that predicts a student’s performance based on daily study habits, sleep, screen time, attendance, previous marks, and assignments completed. The app provides actionable suggestions to help students improve their academic outcomes.

**Technologies Used:**
- Python 3
- Streamlit for web interface
- scikit-learn for Machine Learning (Logistic Regression)
- Pandas & NumPy for data processing
- Joblib for model serialization
- Matplotlib for feature visualization

---

## Features

- Predicts student performance as:
  - **High** — Excellent performance
  - **Needs Improvement** — Moderate performance
  - **At Risk** — Requires immediate improvement
- Provides suggestions to improve study habits and overall performance.
- Visualizes **feature importance** for better understanding.
- User-friendly **interactive web app** using Streamlit.

---

## Dataset

- The dataset is **synthetically generated** to simulate realistic student behavior.
- Features include:
  - `study_hours` — Hours of study per day
  - `sleep_hours` — Hours of sleep per day
  - `screen_time` — Hours of screen time per day
  - `attendance` — Attendance percentage
  - `previous_marks` — Previous marks percentage
  - `assignments_completed` — Number of assignments completed
  - `result` — Target label (`High`, `Needs Improvement`, `At Risk`)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/StudyPulse.git
cd StudyPulse
Create a virtual environment:

2.python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
Install dependencies:

pip install -r requirements.txt
Run Locally
streamlit run app.py
The app will open in your default browser at http://localhost:8501.

4.Deployment
The app can be deployed on Streamlit Cloud.

Create a Streamlit Cloud account, link your GitHub repository, and click Deploy to get a live URL.

5.Model
Model Used: Logistic Regression

Accuracy: ~87%

Serialized using: Joblib (model.pkl, scaler.pkl, label_encoder.pkl)

6.How it Works
1.User enters daily habits and academic details.
2.Inputs are scaled using StandardScaler.
3.Logistic Regression model predicts the performance category.
4.Suggestions are generated based on model prediction and feature importance.

