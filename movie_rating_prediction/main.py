import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from utils import build_pipeline
import matplotlib.pyplot as plt

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/movies.csv", encoding='latin1')

# Drop rows with missing values in important columns
df.dropna(subset=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Year', 'Rating'], inplace=True)

# Extract year as integer (first 4 digits)
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

# Extract duration in minutes (digits only)
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)

# Merge actor columns into a single string
df['Actors'] = df['Actor 1'].str.strip() + ', ' + df['Actor 2'].str.strip() + ', ' + df['Actor 3'].str.strip()

# Define feature columns and target
features = ['Genre', 'Director', 'Actors', 'Duration', 'Year']
target = 'Rating'

X = df[features]
y = df[target]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train the pipeline
pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

# Save the trained model
model_path = "models/movie_rating_model.pkl"
joblib.dump(pipeline, model_path)
print(f"✅ Model saved successfully to '{model_path}'")

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
print("\n📊 Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Predict rating for a sample movie
sample_movie = pd.DataFrame([{
    'Genre': 'Drama',
    'Director': 'Amol Palekar',
    'Actors': 'Rajat Kapoor, Rituparna Sengupta, Antara Mali',
    'Duration': 105,
    'Year': 2010
}])

predicted_rating = pipeline.predict(sample_movie)[0]
print(f"\n🎬 Predicted Rating for sample movie: {predicted_rating:.2f}")

# Plot actual vs predicted ratings
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal perfect prediction line
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Movie Ratings')
plt.grid(True)
plt.show()
