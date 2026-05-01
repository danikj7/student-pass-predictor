import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample training data
# Features: [study hours, attendance %]
X = np.array([
    [2, 40],
    [3, 50],
    [4, 60],
    [5, 70],
    [6, 80],
    [7, 85],
    [8, 90],
    [1, 30],
    [2, 35],
    [9, 95],
])

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
