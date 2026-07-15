# This is a multiple linear regression progression
# Our features include experience,education,certifications,salary

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score 
)
import matplotlib.pyplot as plt
import joblib

# Read the csv
df = pd.read_csv("employee_salary.csv")
# Make sure the data has loaded correctly 
print("Head; \n", df.head()) 
# Check the dataset information
print("Info: \n", df.info())
# Define the dataset features 
X = df[["Experience", "Education", "Certifications"]]
# Define the target (what we want to predict)
y = df["Salary"]

# Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 20
)

# Checking how many samples do we have for each 
print("Training samples: ", len(X_train))
print("Testing samples: ", len(X_test))

# Create the model 
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Check what the model has learned 
print('Intercept: ', model.intercept_)
print("Coefficients: ", model.coef_)

# Make the predictions 
y_pred = model.predict(X_test)

# Compare the actual and predicted values 
results = pd.DataFrame({
    "Actual: ": y_test,
    'Predicted: ': y_pred
})
print(results)

# Find the avg absolute difference between the actual and predicted values
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)

# Find the squared error 
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

# Find the root mean squared error 
rmse = mse ** 0.5 
print("RMSE: ", rmse)

# Find the R2 score 
r2 = r2_score(y_test, y_pred)
print("R2 Score: ", r2)

# Plot the results 
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red"
)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Scores")
plt.show()


joblib.dump(model, "employee_salary_model.pkl")