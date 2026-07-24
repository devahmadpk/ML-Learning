import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("student_pass.csv")

print(df.head())
print(df.info())

# Features and target
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance")
print(importance)

# Predict new student
new_student = pd.DataFrame({
    "Study_Hours":[8],
    "Attendance":[85],
    "Previous_Score":[75]
})

prediction = model.predict(new_student)

print("\nPrediction:", "Pass" if prediction[0] else "Fail")

# Visualize tree
plt.figure(figsize=(14,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail","Pass"],
    filled=True
)

plt.show()

# Save model
joblib.dump(model,"student_pass_model.pkl")