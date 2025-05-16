import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from rich.console import Console
from rich.table import Table
from tabulate import tabulate

# Initialize rich console
console = Console()


# 1. Load the dataset
def load_data(path):
    return pd.read_csv(path)


# 2. Preprocess the dataset
def preprocess_data(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df


# 3. Visualize data
def visualize_data(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.countplot(x='Survived', data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Survival Count")

    sns.countplot(x='Survived', hue='Sex', data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Survival by Gender")

    sns.countplot(x='Survived', hue='Pclass', data=df, ax=axes[1, 0])
    axes[1, 0].set_title("Survival by Class")

    sns.histplot(df['Age'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Age Distribution")

    plt.tight_layout()
    plt.show()


# 4. Train model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


# 5. Evaluate model
def evaluate_model(model, X_test, y_test, X_all, y_all, feature_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cv_scores = cross_val_score(model, X_all, y_all, cv=5)

    coef = model.coef_[0]
    feature_importance = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)

    console.rule("[bold blue]📊 Titanic Survival Prediction Summary")

    console.print(f"[bold green]✅ Model Accuracy:[/bold green] {acc:.2%}")
    console.print(f"[bold green]✅ Cross-Validation Accuracy:[/bold green] {cv_scores.mean():.2%}")

    console.rule("[bold yellow]📌 Confusion Matrix")
    cm_table = Table(show_header=True, header_style="bold magenta")
    cm_table.add_column(" ", style="dim")
    cm_table.add_column("Predicted: 0")
    cm_table.add_column("Predicted: 1")
    cm_table.add_row("Actual: 0", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("Actual: 1", str(cm[1][0]), str(cm[1][1]))
    console.print(cm_table)

    console.rule("[bold yellow]📌 Classification Report")
    console.print(report)

    console.rule("[bold yellow]📌 Feature Importance (by Coefficient)")
    table_str = tabulate(feature_importance, headers=["Feature", "Coefficient"], floatfmt=".4f")
    console.print(table_str)


# 6. Main function
def main():
    df = load_data("titanic.csv")
    console.print("[bold cyan]✅ Data Loaded Successfully!\n")

    console.rule("[bold yellow]🔍 Missing Values Before Cleaning")
    missing = df.isnull().sum()
    print(missing[missing > 0], "\n")

    visualize_data(df)

    df = preprocess_data(df)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, X_scaled, y, features)

    joblib.dump(model, "titanic_model.pkl")
    console.print("\n💾 [bold green]Model saved as 'titanic_model.pkl'[/bold green]")


# Run the script
if __name__ == "__main__":
    main()
