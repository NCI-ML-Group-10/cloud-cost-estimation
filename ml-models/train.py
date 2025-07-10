"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-19 21:55:40
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-20 19:21:53
FilePath: /cloud-cost-estimation/ml-models/train.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from preprocessing import build_preprocessor, load_csv_data
from clearml import Task

task = Task.init(project_name="NCI-ML-Project", task_name="cost_training")

df = load_csv_data()

preprocessor, features, target = build_preprocessor()

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# build pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10)),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

task.get_logger().report_scalar("RMSE", "RMSE", iteration=0, value=rmse)
task.get_logger().report_scalar("R2", "R2", iteration=0, value=r2)
print(f"✅ Done. RMSE: {rmse:.2f}, R2: {r2:.4f}")

errors = np.abs(y_pred.flatten() - y_test)
# ✅ 使用箱线图展示误差分布
task.get_logger().report_histogram(
    title="Prediction Error",
    series="Absolute Error",
    values=errors.tolist(),
    iteration=0,
)

import matplotlib.pyplot as plt

sorted_indices = np.argsort(y_test)
y_test_sorted = np.array(y_test)[sorted_indices]
y_pred_sorted = np.array(y_pred)[sorted_indices]

plt.figure(figsize=(8, 4))
plt.plot(y_test_sorted, label="Actual", linewidth=2)
plt.plot(y_pred_sorted, label="Predicted", linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Cloud Cost")
plt.legend()
plt.grid(True)

# ✅ 上传到 ClearML
task.get_logger().report_matplotlib_figure(
    title="Prediction vs Actual (Line Plot)",
    series="Cost Curve",
    figure=plt.gcf(),
    iteration=0,
)
plt.close()
