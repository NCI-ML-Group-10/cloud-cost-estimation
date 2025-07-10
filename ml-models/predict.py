"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-19 22:18:14
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-20 19:24:35
FilePath: /cloud-cost-estimation/ml-models/predict.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import build_preprocessor, load_csv_data

df = load_csv_data()

preprocessor, features, target = build_preprocessor()

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# load model locally
# model_uri = "runs:/c5d2f1683818446987175ab977dcd0c4/model"
model = mlflow.sklearn.load_model("local_model")

# doing predict
y_pred = model.predict(X_test)
print("Prediction Preview:", y_pred[:5])

# visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label="Actual", marker="o")
plt.plot(y_pred[:50], label="Predicted", marker="x")
plt.title("Actual vs Predicted GCP Cost")
plt.xlabel("Sample Index")
plt.ylabel("Total Cost (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_prediction_plot.png")
