# This is a polynomial regression project
# Our features include car engine size  

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import joblib

# Read the csv
df = pd.read_csv("fuel_efficiency.csv")
# Make sure the data has loaded correctly 
print("Head: \n", df.head())
# Check the dataset information
print("Info: \n", df.info())
# Define the dataset features 
X = df[["Engine_Size"]] 
# Define the target (what we want to predict)
y = df["MPG"] 

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size = 0.2,
    random_state = 20
)

# Create polynomial features
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create the model
model = LinearRegression()

# Train the model 
model.fit(X_train_poly, y_train)

# View the learned parameters
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Make predictions
y_pred = model.predict(X_test_poly)

# Compare the results with the actual data
results = pd.DataFrame({
    "Actual MPG: ": y_test,
    "Predicted MPG: ": y_pred
})

print(results)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# Plot the original data
plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", label="Actual Data")

# Create smooth curve
X_curve = pd.DataFrame({
    "Engine_Size": np.linspace(
        X["Engine_Size"].min(),
        X["Engine_Size"].max(),
        200
    )
})

X_curve_poly = poly.transform(X_curve)

y_curve = model.predict(X_curve_poly)

plt.plot(
    X_curve,
    y_curve,
    color="red",
    linewidth=2,
    label="Polynomial Regression"
)

plt.xlabel("Engine_Size")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

# Predict a new house price
new_efficiency = pd.DataFrame({
    "Engine_Size": [2.7]
    })

new_efficiency_poly = poly.transform(new_efficiency)

prediction = model.predict(new_efficiency_poly)

print(f"\nPredicted Efficiency: {prediction[0]:.2f}")

# Save the model
joblib.dump(model, "car_efficiency_model.pkl")
joblib.dump(poly, "polynomial_transformer.pkl")