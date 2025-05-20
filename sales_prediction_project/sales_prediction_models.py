import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('advertising.csv')

# Check data
print("\n✅ Dataset Loaded Successfully")
print(df.head())
print("\n📊 Dataset Info:")
print(df.info())
print("\n🔍 Checking for Missing Values:")
print(df.isnull().sum())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Prepare features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

print("\n🔎 Model Comparison:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} => MSE: {mse:.2f}, R²: {r2:.4f}")

# Train best model and save
best_model = LinearRegression()
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'sales_model.pkl')

# Feature Importance (from Random Forest)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()