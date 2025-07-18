"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-07-10 17:16:51
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-07-10 17:16:54
FilePath: /cloud-cost-estimation/ml-models/baseline.py
Description:
base line model diff
Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from clearml import Task, Dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

# keep local version is same with online environment
Task.add_requirements("scikit-learn", "1.7.0")

# init the ClearML task
task = Task.init(
    project_name="NCI-ML-Project",
    task_name="Bagging Tree Regressor Training",
    task_type=Task.TaskTypes.training,
    output_uri=True,
)

logger = task.get_logger()

# read dataset from the clearml
dataset = Dataset.get(
    dataset_project="NCI-ML-Project", dataset_name="Gcp_Cloud_Billing"
)
dataset_path = dataset.get_local_copy()
csv_path = os.path.join(dataset_path, "gcp_final_approved_dataset.csv")
df = pd.read_csv(csv_path)

# feature engineering
df["Service_Avg_Cost"] = (
    df.groupby("Service Name")["Total Cost (INR)"].transform("mean").fillna(0)
)
df["Region_Zone_Avg_Cost"] = (
    df.groupby("Region/Zone")["Total Cost (INR)"].transform("mean").fillna(0)
)
df["Usage Start Date"] = pd.to_datetime(
    df["Usage Start Date"], format="%d-%m-%Y %H:%M", errors="coerce"
)
df["Usage End Date"] = pd.to_datetime(
    df["Usage End Date"], format="%d-%m-%Y %H:%M", errors="coerce"
)
df["start_hour"] = df["Usage Start Date"].dt.hour
df["start_day_of_week"] = df["Usage Start Date"].dt.dayofweek
df["start_month"] = df["Usage Start Date"].dt.month
df["is_weekend"] = df["start_day_of_week"].isin([5, 6]).astype(int)
df["is_night_usage"] = df["start_hour"].apply(lambda h: 1 if h <= 6 or h >= 22 else 0)
df["duration_minutes"] = (
    df["Usage End Date"] - df["Usage Start Date"]
).dt.total_seconds() / 60
df["duration_minutes"] = df["duration_minutes"].fillna(0)

# feature define
cat_cols = ["Service Name", "Region/Zone"]
num_cols = [
    "Usage Quantity",
    "CPU Utilization (%)",
    "Memory Utilization (%)",
    "Network Inbound Data (Bytes)",
    "Network Outbound Data (Bytes)",
    "Cost per Quantity ($)",
    "Service_Avg_Cost",
    "Region_Zone_Avg_Cost",
    "start_hour",
    "start_day_of_week",
    "start_month",
    "is_weekend",
    "is_night_usage",
    "duration_minutes",
]
target = "Rounded Cost ($)"
features = cat_cols + num_cols
X = df[features]
y = df[target]

# transformation
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)
# build pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),
                n_estimators=30,
                max_samples=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# traing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Baseline 1: DummyRegressor
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
rmse_dummy = np.sqrt(mean_squared_error(y_test, y_pred_dummy))
r2_dummy = r2_score(y_test, y_pred_dummy)
mae_dummy = mean_absolute_error(y_test, y_pred_dummy)

# Baseline 2: LinearRegression
lr = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Log baseline performance to ClearML
logger.report_scalar("metrics", "RMSE_Dummy", value=rmse_dummy, iteration=10)
logger.report_scalar("metrics", "R2_Dummy", value=r2_dummy, iteration=11)
logger.report_scalar("metrics", "MAE_Dummy", value=mae_dummy, iteration=12)
logger.report_scalar("metrics", "RMSE_LR", value=rmse_lr, iteration=20)
logger.report_scalar("metrics", "R2_LR", value=r2_lr, iteration=21)
logger.report_scalar("metrics", "MAE_LR", value=mae_lr, iteration=22)

# display summary
print(f"BaggingRegressor: RMSE={rmse:.2f}, R2={r2:.4f}, MAE={mae:.2f}")
print(f"DummyRegressor:  RMSE={rmse_dummy:.2f}, R2={r2_dummy:.4f}, MAE={mae_dummy:.2f}")
print(f"LinearRegression: RMSE={rmse_lr:.2f}, R2={r2_lr:.4f}, MAE={mae_lr:.2f}")

# all model result comparison
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", linewidth=2, color="black")
plt.plot(y_pred, label="BaggingRegressor", linewidth=2)
plt.plot(y_pred_lr, label="LinearRegression", linestyle="--", linewidth=2)
plt.plot(y_pred_dummy, label="DummyRegressor", linestyle=":", linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Cloud Cost")
plt.title("Prediction Comparison: Actual vs. Models")
plt.legend()
plt.grid(True)

# upload comparison figure to clearml
task.get_logger().report_matplotlib_figure(
    title="Prediction vs Actual (Comparison)",
    series="Result Comparison",
    figure=plt.gcf(),
    iteration=5,
)
plt.close()

# ml algorithm name and metrics
model_names = ["BaggingRegressor", "DummyRegressor", "LinearRegression"]
rmse_values = [rmse, rmse_dummy, rmse_lr]
mae_values = [mae, mae_dummy, mae_lr]
r2_values = [r2, r2_dummy, r2_lr]

x = np.arange(len(model_names))
bar_width = 0.22

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - bar_width, rmse_values, width=bar_width, label="RMSE")
bars2 = ax.bar(x, mae_values, width=bar_width, label="MAE")
bars3 = ax.bar(x + bar_width, [v * 100 for v in r2_values], width=bar_width, label="R²")

# add value tag
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        if bars is bars3:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylabel("Metric Value (RMSE/MAE, R² x100)")
ax.set_title("Baseline Model Comparison: RMSE, MAE, R²")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()

# upload to the clearml
task.get_logger().report_matplotlib_figure(
    title="Baseline Model Comparison (RMSE, MAE, R2)",
    series="Bar Chart",
    figure=plt.gcf(),
    iteration=6,
)
plt.savefig("baseline_model.png")
plt.close()
