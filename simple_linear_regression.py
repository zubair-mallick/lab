import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
hours_studied = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
exam_score = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120, 130])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    hours_studied.reshape(-1, 1),
    exam_score,
    test_size=0.2,
    random_state=42
)

# Create a Simple Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.2f}')
