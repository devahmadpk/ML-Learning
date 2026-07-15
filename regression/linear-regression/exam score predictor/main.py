# This is a multiple linear regression progression
# Our features include hours studied, sleep hours and attendance 

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score 
)
import joblib

# Read the csv
df = pd.read_csv("student_scores.csv")
# Make sure data has loaded correctly 
print("Head: \n", df.head())  
# Check dataset information
print("Info: \n", df.info())
# Define dataset features
X = df[["Hours", "Sleep", "Attendance"]]
# Define the target (what we want to predict)
y = df["Score"]

# Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2, # 20% data kept for testing 
    random_state = 42
)

# Checking how many samples do we have for each 
print("Training samples: ", len(X_train))
print("Testing samples: ", len(X_test))

# Create the model 
model = LinearRegression()

# Train the model 
model.fit(X_train, y_train)

# Check what the model has learned 
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_) 

# Make predictions
y_pred = model.predict(X_test)

# Compare actual and predicted values 
results = pd.DataFrame({
    "Actual: ": y_test,
    "Predicted: ": y_pred
})
print(results )

# Find the avg absolute difference between the actual and predicted values 
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)

# Find the squared error 
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

# Find the root mean squared error 
rmse = mean_squared_error(y_test, y_pred) ** 0.5 
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
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores")
plt.show()


joblib.dump(model, "student_score_model.pkl")