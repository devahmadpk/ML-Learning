# Diabetes Prediction using Logistic Regression

A beginner-friendly Machine Learning project that predicts whether a patient is **Diabetic** or **Not Diabetic** using **Logistic Regression**.

The model uses three input features:

- Glucose Level
- Body Mass Index (BMI)
- Age

to predict whether a patient has diabetes.

---

## Project Objective

The goal of this project is to understand and implement the complete Machine Learning workflow using **Logistic Regression** for binary classification.

Instead of predicting a continuous value, this project predicts one of two classes:

- Not Diabetic (0)
- Diabetic (1)

The project covers every stage of the Machine Learning workflow including data preparation, model training, prediction, evaluation, probability estimation, visualization, and model persistence.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Glucose | Blood glucose level |
| BMI | Body Mass Index |
| Age | Patient's age |
| Diabetes | Target Variable (0 = Not Diabetic, 1 = Diabetic) |

Example:

| Glucose | BMI | Age | Diabetes |
|---------:|----:|----:|----------:|
|85|22.1|25|0|
|110|31.2|45|1|
|140|38.1|60|1|

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
6. Predict Patient Classes
7. Evaluate Model Accuracy
8. Predict Class Probabilities
9. Visualize Dataset
10. Predict Diabetes for a New Patient
11. Save Trained Model

---

## Model Evaluation Metrics

The following metric is used to evaluate the model:

- Accuracy Score

> More advanced classification metrics such as Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and ROC-AUC will be covered in future projects.

---

## Prediction Probabilities

Unlike regression models, Logistic Regression predicts **probabilities** for each class.

Example:

| Probability of No Diabetes | Probability of Diabetes | Prediction |
|---------------------------:|------------------------:|-----------|
|0.94|0.06|Not Diabetic|
|0.18|0.82|Diabetic|

The model predicts the class with the higher probability.

---

## Visualization

The project visualizes:

- Glucose Level vs BMI
- Diabetic and Non-Diabetic Patients

This helps visualize how the two classes are distributed within the dataset.

---

## Project Structure

```text
Diabetes-Prediction/
│
├── diabetes.csv
├── main.py
├── diabetes_model.pkl
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
- Making Class Predictions
- Predicting Class Probabilities
- Model Evaluation using Accuracy
- Data Visualization
- Saving Trained Models with Joblib

---

## Future Improvements

Future versions of this project can include:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC Curve
- ROC-AUC Score
- Decision Boundary Visualization
- Feature Scaling
- Hyperparameter Tuning
- Cross Validation