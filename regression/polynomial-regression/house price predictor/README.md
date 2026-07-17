# House Price Prediction using Polynomial Regression

A beginner-friendly Machine Learning project that predicts the price of a house using Polynomial Regression.

The model uses one input feature:

- House Size (Square Feet)

to predict the selling price of a house.

---

## Project Objective

The goal of this project is to understand and implement Polynomial Regression for solving non-linear regression problems.

This project demonstrates how Linear Regression can model curved relationships by generating polynomial features from the original input feature.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Size | House size in square feet |
| Price | House price (Target Variable) |

Example:

| Size | Price |
|------|--------|
|500|50000|
|700|70000|
|900|95000|

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
5. Generate Polynomial Features
6. Train Polynomial Regression Model
7. Evaluate Model
8. Visualize Regression Curve
9. Predict New House Prices
10. Save Trained Model and Polynomial Transformer

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

- Original House Price Data
- Polynomial Regression Curve

This helps determine how well the model captures the non-linear relationship between house size and price.

---

## Project Structure

```text
House Price Predictor/
│
├── house_prices.csv
├── main.py
├── house_price_model.pkl
├── polynomial_transformer.pkl
└── README.md
```

---

## Key Concepts Learned

- Polynomial Regression
- Feature Engineering
- Polynomial Features
- Non-linear Relationships
- Train-Test Split
- Model Evaluation
- Regression Metrics
- Data Visualization
- Model Persistence


---

## Future Improvements

- Compare multiple polynomial degrees.
- Evaluate the model using Cross Validation.
- Compare Polynomial Regression with Linear Regression.
- Experiment with Ridge and Lasso Regression.