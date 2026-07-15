# Employee Salary Prediction using Multiple Linear Regression

A beginner-friendly Machine Learning project that predicts an employee's salary using Multiple Linear Regression.

The model uses three input features:

- Years of Experience
- Years of Education
- Number of Professional Certifications

to predict their annual salary.

---

## Project Objective

The goal of this project is to understand and implement the complete Machine Learning workflow using Multiple Linear Regression.

Instead of simply training a model, this project covers every stage involved in building a regression model, including data preparation, training, evaluation, visualization, prediction, and model persistence.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Experience | Years of work experience |
| Education | Years of education completed |
| Certifications | Number of professional certifications |
| Salary | Annual salary (Target Variable) |

Example:

| Experience | Education | Certifications | Salary |
|------------|-----------|----------------|---------|
|1|12|0|35000|
|5|16|2|68000|
|10|18|5|112000|

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

he project visualizes:

- Actual Salary vs Predicted Salary
- Perfect Prediction Line

This helps determine how closely the model's predictions match the actual values.

## Project Structure

```
Employee-Salary-Predictor/
│
├── employee_salary.csv
├── employee_salary.py
├── employee_salary_model.pkl
└── README.md
```