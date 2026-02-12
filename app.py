import streamlit as st
import joblib
import numpy as np

# Load saved model files
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 StudyPulse - Student Performance Predictor")

st.write("Enter student details below:")

# User inputs
study_hours = st.number_input("Study Hours per day", min_value=1, max_value=12)
sleep_hours = st.number_input("Sleep Hours per day", min_value=1, max_value=12)
screen_time = st.number_input("Screen Time per day", min_value=0, max_value=12)
attendance = st.number_input("Attendance Percentage", min_value=0, max_value=100)
previous_marks = st.number_input("Previous Marks (%)", min_value=0, max_value=100)
assignments_completed = st.number_input("Assignments Completed (out of 10)", min_value=0, max_value=10)

if st.button("Predict Performance"):

    # Create distraction score
    distraction_score = screen_time / study_hours if study_hours != 0 else 0

    # Prepare input data
    input_data = np.array([[study_hours, sleep_hours, screen_time,
                            attendance, previous_marks,
                            assignments_completed, distraction_score]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction)

    # Confidence
    probabilities = model.predict_proba(input_data)
    confidence = np.max(probabilities) * 100

    st.subheader("Prediction Result:")
    st.success(f"{prediction_label[0]} (Confidence: {confidence:.2f}%)")

    # Improvement suggestions
    if prediction_label[0] == "At Risk":
        st.warning("⚠ Suggestions:")
        st.write("- Increase study hours")
        st.write("- Reduce screen time")
        st.write("- Improve attendance")
    elif prediction_label[0] == "Needs Improvement":
        st.info("📌 Suggestions:")
        st.write("- Slightly increase study time")
        st.write("- Maintain consistent sleep")
    else:
        st.success("🌟 Excellent performance! Keep it up!")
