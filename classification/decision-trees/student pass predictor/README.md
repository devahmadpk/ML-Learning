# Student Pass Prediction using Decision Tree Classification

A beginner-friendly Machine Learning project that predicts whether a student will pass or fail using a Decision Tree Classifier.

The model uses three input features:

- Study Hours
- Attendance
- Previous Score

to predict whether a student will pass or fail.

---

## Project Objective

The goal of this project is to understand and implement the complete Machine Learning workflow using a Decision Tree Classifier.

Instead of simply training a model, this project covers every stage involved in building a classification model, including data preparation, training, evaluation, visualization, prediction, feature importance analysis, and model persistence.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Study_Hours | Number of hours the student studied |
| Attendance | Attendance percentage |
| Previous_Score | Previous exam score |
| Pass | Target Variable (0 = Fail, 1 = Pass) |

Example:

| Study_Hours | Attendance | Previous_Score | Pass |
|--------------|------------|----------------|------|
|2|60|45|0|
|6|75|65|1|
|10|90|85|1|

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-Learn
- Joblib

---

## Machine Learning Workflow

1. Load Dataset
2. Explore Dataset
3. Define Features and Target
4. Split Dataset into Training and Testing Data
5. Train Decision Tree Classifier
6. Evaluate Model
7. Visualize the Decision Tree
8. Predict New Student Result
9. Analyze Feature Importance
10. Save Trained Model

---

## Model Evaluation Metrics

The following metrics are used to evaluate the model:

- Accuracy
- Confusion Matrix
- Classification Report

---

## Visualization

The project visualizes:

- Complete Decision Tree
- Decision Nodes
- Leaf Nodes
- Splitting Conditions

This helps understand how the Decision Tree makes decisions.

---

## Project Structure

```text
Student-Pass-Predictor/
│
├── student_pass.csv
├── main.py
├── student_pass_model.pkl
└── README.md
```