# iris_classifier.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Explore data
print("Dataset Preview:\n", df.head())
print("\nClass Distribution:\n", df['species'].value_counts())

# Enhanced Data Visualization
sns.set(style="whitegrid")
pairplot = sns.pairplot(df, hue='species', palette='Set2', markers=["o", "s", "D"])
pairplot.fig.suptitle("Iris Dataset - Sepal & Petal Analysis", y=1.02, fontsize=16)
pairplot.fig.set_size_inches(10, 8)

# Save the figure
pairplot.savefig("iris_pairplot.png")
plt.show()

# Feature and target split
X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save the best model
joblib.dump(best_model, 'iris_model.pkl')
print(f"\n✅ Best model saved as 'iris_model.pkl' with accuracy: {best_accuracy:.2f}")

# Predict with new sample using DataFrame with feature names to avoid warning
sample_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
prediction = best_model.predict(sample_input)
print(f"\n🔍 Predicted class for {sample_input.values.tolist()}: {prediction[0]}")
