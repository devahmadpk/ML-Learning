# This is a Logistic Regression project
# Our features include glucose level, BMI and age

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Read the csv
df = pd.read_csv("diabetes.csv")

# Make sure the data has loaded correctly
print("Head:\n", df.head())

# Check the dataset information
print("Info:\n", df.info())

# Define the dataset features
X = df[["Glucose", "BMI", "Age"]]

# Define the target (what we want to predict)
y = df["Diabetes"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=20
)

# Check how many samples we have
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# View what the model learned
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions
y_pred = model.predict(X_test)

# Compare actual and predicted values
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(results)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")

# Predict probabilities
probabilities = model.predict_proba(X_test)

probability_results = pd.DataFrame(
    probabilities,
    columns=[
        "Probability of No Diabetes",
        "Probability of Diabetes"
    ]
)

print("\nPrediction Probabilities")
print(probability_results)

# Predict a new patient
new_patient = pd.DataFrame({
    "Glucose": [122],
    "BMI": [33.8],
    "Age": [49]
})

prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print("\nNew Patient Prediction")

if prediction[0] == 1:
    print("Prediction: Diabetic")
else:
    print("Prediction: Not Diabetic")

print(f"Probability of Diabetes: {probability[0][1]:.2%}")

# Visualize the dataset
plt.figure(figsize=(8,6))

colors = ["blue" if value == 0 else "red" for value in y]

plt.scatter(
    df["Glucose"],
    df["BMI"],
    c=colors,
    s=80
)

plt.xlabel("Glucose Level")
plt.ylabel("BMI")
plt.title("Diabetes Prediction Dataset")
plt.grid(True)

plt.show()

# Save the trained model
joblib.dump(model, "diabetes_model.pkl")