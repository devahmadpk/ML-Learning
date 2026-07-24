import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("house_prices.csv")

print(df.head())
print(df.info())

# Features and target
X = df[["Size","Bedrooms","Age"]]
y = df["Price"]

# Split dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

# Train model
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse**0.5
r2 = r2_score(y_test,y_pred)

print("MAE :",mae)
print("MSE :",mse)
print("RMSE:",rmse)
print("R²  :",r2)

# Compare results
results = pd.DataFrame({
    "Actual":y_test,
    "Predicted":y_pred
})

print(results)

# Feature importance
importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

print("\nFeature Importance")
print(importance)

# Predict new house
new_house = pd.DataFrame({
    "Size":[2400],
    "Bedrooms":[4],
    "Age":[3]
})

prediction = model.predict(new_house)

print("\nPredicted Price:",prediction[0])

# Visualize tree
plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)

plt.show()

# Save model
joblib.dump(model,"house_price_model.pkl")