# House Price Prediction using Decision Tree Regression

A beginner-friendly Machine Learning project that predicts house prices using a Decision Tree Regressor.

The model uses three input features:

- House Size
- Number of Bedrooms
- House Age

to predict the selling price of a house.

---

## Project Objective

The goal of this project is to understand and implement the complete Machine Learning workflow using a Decision Tree Regressor.

Instead of simply training a model, this project covers every stage involved in building a regression model, including data preparation, training, evaluation, visualization, prediction, feature importance analysis, and model persistence.

---

## Dataset

The dataset contains the following columns:

| Feature | Description |
|----------|-------------|
| Size | House size (square feet) |
| Bedrooms | Number of bedrooms |
| Age | Age of the house (years) |
| Price | Selling price (Target Variable) |

Example:

| Size | Bedrooms | Age | Price |
|------|----------|-----|--------|
|850|2|18|145000|
|1700|4|6|325000|
|3100|6|1|720000|

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
5. Train Decision Tree Regressor
6. Evaluate Model
7. Visualize the Decision Tree
8. Predict New House Price
9. Analyze Feature Importance
10. Save Trained Model

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

- Complete Decision Tree
- Decision Nodes
- Leaf Nodes
- Splitting Conditions

This helps understand how the Decision Tree makes predictions.

---

## Project Structure

```text
House-Price-Predictor/
│
├── house_prices.csv
├── main.py
├── house_price_model.pkl
└── README.md
```