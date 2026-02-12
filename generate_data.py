import pandas as pd
import random
import numpy as np

data = []
scores = []

# First generate raw data + score
for _ in range(500):

    study_hours = random.randint(1, 12)
    sleep_hours = random.randint(4, 9)
    screen_time = random.randint(0, 8)
    attendance = random.randint(50, 100)
    previous_marks = random.randint(40, 95)
    assignments_completed = random.randint(1, 10)

    distraction_ratio = screen_time / (study_hours + 1)

    score = (
        previous_marks * 0.35
        + attendance * 0.25
        + study_hours * 3
        + assignments_completed * 2.5
        + sleep_hours * 1.5
        - distraction_ratio * 10
    )

    score += random.uniform(-8, 8)

    scores.append(score)

    data.append([
        study_hours,
        sleep_hours,
        screen_time,
        attendance,
        previous_marks,
        assignments_completed,
        score
    ])

# Convert to DataFrame
columns = [
    "study_hours",
    "sleep_hours",
    "screen_time",
    "attendance",
    "previous_marks",
    "assignments_completed",
    "score"
]

df = pd.DataFrame(data, columns=columns)

# Use percentiles for balanced classes
low_threshold = np.percentile(df["score"], 33)
high_threshold = np.percentile(df["score"], 66)

def classify(score):
    if score >= high_threshold:
        return "High"
    elif score >= low_threshold:
        return "Needs Improvement"
    else:
        return "At Risk"

df["result"] = df["score"].apply(classify)

# Drop score column
df = df.drop("score", axis=1)

df.to_csv("data/student_data.csv", index=False)

print(df["result"].value_counts())
print("Balanced realistic dataset generated successfully!")
