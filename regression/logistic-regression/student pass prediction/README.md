# Student Pass Prediction using Logistic Regression

A beginner-friendly Machine Learning project that predicts whether a student will **Pass** or **Fail** using **Logistic Regression**.

The model uses two input features:

- Hours Studied
- Attendance Percentage

to predict whether a student passes the exam.

---

## Project Objective

The goal of this project is to understand and implement the complete Machine Learning workflow using **Logistic Regression** for binary classification.

Instead of predicting a continuous value, this project predicts one of two classes:

- Pass (1)
- Fail (0)

The project covers every stage of the Machine Learning workflow including data preparation, model training, prediction, evaluation, probability estimation, visualization, and model persistence.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Hours | Number of hours studied |
| Attendance | Attendance percentage |
| Pass | Target Variable (0 = Fail, 1 = Pass) |

Example:

| Hours | Attendance | Pass |
|-------:|-----------:|-----:|
|1|50|0|
|4|72|0|
|6|82|1|
|8|92|1|

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
5. Train Logistic Regression Model
6. Predict Student Classes
7. Evaluate Model Accuracy
8. Predict Class Probabilities
9. Visualize Dataset
10. Predict New Student Result
11. Save Trained Model

---

## Model Evaluation Metrics

The following metric is used to evaluate the model:

- Accuracy Score

> More advanced classification metrics such as Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC will be covered in future projects.

---

## Prediction Probabilities

Unlike Linear Regression, Logistic Regression predicts **probabilities**.

Example:

| Probability of Fail | Probability of Pass | Prediction |
|--------------------:|--------------------:|-----------|
|0.92|0.08|Fail|
|0.15|0.85|Pass|

The model predicts the class with the higher probability.

---

## Visualization

The project visualizes:

- Student Dataset
- Pass and Fail Classes

This provides a visual understanding of how the data is distributed before classification.

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

---

## Learning Outcomes

By completing this project, you will learn:

- Binary Classification
- Logistic Regression
- Feature and Target Selection
- Train-Test Split
- Model Training
- Making Predictions
- Predicting Probabilities
- Model Evaluation using Accuracy
- Saving Trained Models with Joblib

---

## Future Improvements

Future versions of this project can include:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score
- Decision Boundary Visualization
- Feature Scaling
- Hyperparameter Tuning
- Cross Validation