# Student Score Prediction using Multiple Linear Regression

A beginner-friendly Machine Learning project that predicts a student's exam score using Multiple Linear Regression.

The model uses three input features:

- Hours Studied
- Sleep Hours
- Attendance Percentage

to predict the student's final exam score.

---

## Project Objective

The objective of this project is to understand the complete workflow of a supervised Machine Learning regression problem.

Instead of simply training a model, this project covers every stage involved in building a regression model, including data preparation, training, evaluation, visualization, prediction, and model persistence.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Hours | Number of hours studied |
| Sleep | Number of hours slept |
| Attendance | Attendance percentage |
| Score | Final exam score (Target Variable) |

Example:

| Hours | Sleep | Attendance | Score |
|-------|-------|------------|------|
|1|6|60|42|
|2|6|65|48|
|3|7|80|63|

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
5. Train Multiple Linear Regression Model
6. Evaluate Model
7. Visualize Predictions
8. Predict New Student Scores
9. Save Trained Model

---

## Model Evaluation Metrics

The following metrics are used to evaluate the model:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Visualization

The project visualizes:

- Actual Scores vs Predicted Scores
- Perfect Prediction Line

This helps determine how closely the model's predictions match the actual values.

---

## Project Structure

```
Student-Score-Predictor/
│
├── student_scores.csv
├── main.py
├── student_score_model.pkl
├── README.md
```
