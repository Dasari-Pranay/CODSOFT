# Iris Flower Classification 🌸

This project uses machine learning to classify Iris flowers into:
- Setosa
- Versicolor
- Virginica

based on their sepal and petal length/width.

## 🚀 Features
- Loads and visualizes Iris dataset
- Compares multiple ML models
- Saves the best-performing model
- Predicts new inputs with the trained model

## 📁 Files
- `iris_classifier.py`: Main code
- `iris_model.pkl`: Saved ML model
- `requirements.txt`: Python dependencies

## 📊 Algorithms Used
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest

## 🛠️ How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the script:
    ```bash
    python iris_classifier.py
    ```

3. You’ll see model accuracy, prediction results, and visualizations.

---

## ✅ Sample Prediction

```python
sample_input = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample_input)
