import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data
data = {
    "Customer_Segment": ["Gold", "Silver", "Bronze", "Gold", "Silver", "Bronze"],
    "Product_Category": ["Electronics", "Clothing", "Food", "Home Decor", "Clothing", "Electronics"],
    "Purchase_Amount": [50, 15, 80, 30, 20, 10],
    "Promotion_Used": ["Yes", "No", "No", "Yes", "Yes", "No"],
    "Sales_Target": ["High", "Medium", "Low", "Medium", "Medium", "Low"]
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical codes
df["Customer_Segment"] = pd.Categorical(df["Customer_Segment"]).codes
df["Product_Category"] = pd.Categorical(df["Product_Category"]).codes
df["Promotion_Used"] = pd.Categorical(df["Promotion_Used"]).codes

# Separate features (X) and target variable (y)
X = df[["Customer_Segment", "Product_Category", "Purchase_Amount", "Promotion_Used"]]
y = df["Sales_Target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Print Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report with zero_division=0 to avoid warnings
print(classification_report(y_test, y_pred, zero_division=0))
