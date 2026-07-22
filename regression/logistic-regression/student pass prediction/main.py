# Student Pass/Fail Prediction using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv("student_pass.csv")

# Inspect the dataset
print("Head:\n", df.head())
print("\nInfo:")
print(df.info())

# Define features (inputs)
X = df[["Hours", "Attendance"]]

# Define target (output)
y = df["Pass"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=20
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# View learned parameters
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions
y_pred = model.predict(X_test)

# Compare Actual vs Predicted
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nResults")
print(results)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")

# Predict probability of each class
probabilities = model.predict_proba(X_test)

print("\nPrediction Probabilities")
print(pd.DataFrame(
    probabilities,
    columns=["Probability of Fail", "Probability of Pass"]
))

# Predict a new student
new_student = pd.DataFrame({
    "Hours": [6],
    "Attendance": [85]
})

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("\nNew Student Prediction")

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")

print(f"Probability of Passing: {probability[0][1]:.2%}")

# Scatter plot
plt.figure(figsize=(8,6))

colors = ["red" if value == 0 else "green" for value in y]

plt.scatter(
    df["Hours"],
    df["Attendance"],
    c=colors,
    s=80
)

plt.xlabel("Hours Studied")
plt.ylabel("Attendance (%)")
plt.title("Student Pass/Fail Dataset")
plt.grid(True)

plt.show()

# Save the model
joblib.dump(model, "student_pass_model.pkl")