import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Step 1: Load dataset
print("\n🔹 Loading dataset...")
df = pd.read_csv('creditcard.csv')
print(f"✅ Dataset loaded. Total transactions: {len(df)}")

# Step 2: Check class balance
class_counts = df['Class'].value_counts()
print("\n📊 Class distribution:")
print(f"   Genuine: {class_counts[0]}")
print(f"   Fraudulent: {class_counts[1]}")

# Step 3: Preprocess
print("\n🔹 Preprocessing data...")
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Normalize amount
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("✅ Data split into training and testing sets.")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")

# Step 4: Train LightGBM
print("\n🔹 Training LightGBM Classifier...")
clf = LGBMClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train, y_train)
print("✅ LightGBM training complete.")

# Step 5: Predict & evaluate
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=["Genuine", "Fraudulent"], output_dict=True)

# Step 6: Print results
print("\n📌 CONFUSION MATRIX (ASCII Style)")
print("+------------+-----------+-----------+")
print("|            | Pred 0    | Pred 1    |")
print("+------------+-----------+-----------+")
print(f"| Actual 0   | {cm[0][0]:<9} | {cm[0][1]:<9} |")
print(f"| Actual 1   | {cm[1][0]:<9} | {cm[1][1]:<9} |")
print("+------------+-----------+-----------+")

print("\n📌 CLASSIFICATION METRICS")
print(f"Precision (Fraudulent):  {cr['Fraudulent']['precision']:.4f}")
print(f"Recall (Fraudulent):     {cr['Fraudulent']['recall']:.4f}")
print(f"F1-score (Fraudulent):   {cr['Fraudulent']['f1-score']:.4f}")

# Step 7: Predict probabilities for the positive class
y_proba = clf.predict_proba(X_test)[:, 1]

# Step 8: Threshold-based Precision-Recall Bar Chart
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Select some thresholds evenly spaced between min and max of thresholds
selected_thresholds = np.linspace(0, 1, 6)

prec_at_thresh = []
rec_at_thresh = []

for t in selected_thresholds:
    # Find closest threshold index
    idx = (np.abs(thresholds - t)).argmin()
    prec_at_thresh.append(precisions[idx])
    rec_at_thresh.append(recalls[idx])

plt.figure(figsize=(8,4))
width = 0.35
x = np.arange(len(selected_thresholds))

plt.bar(x - width/2, prec_at_thresh, width, label='Precision', color='royalblue')
plt.bar(x + width/2, rec_at_thresh, width, label='Recall', color='darkorange')

plt.xticks(x, [f"{t:.2f}" for t in selected_thresholds])
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Precision & Recall at Different Thresholds')
plt.legend()
plt.tight_layout()
plt.savefig('precision_recall_thresholds.png')
plt.show()

print("\n📁 Precision-Recall bar chart saved as 'precision_recall_thresholds.png'")
