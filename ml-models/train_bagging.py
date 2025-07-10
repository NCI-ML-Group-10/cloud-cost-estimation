import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from clearml import Task, Dataset, Logger, OutputModel

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# keep local version is same with online environment
Task.add_requirements("scikit-learn", "1.7.0")

# init the ClearML task
task = Task.init(
    project_name="NCI-ML-Project",
    task_name="Bagging Tree Regressor Training",
    task_type=Task.TaskTypes.training,
    output_uri=True,
)

# ✅ 绑定参数（这些参数会被 HPO 动态注入）
params = {
    "regressor__n_estimators": 30,
    "regressor__max_samples": 80,
    "regressor__estimator__max_depth": 3,
}
params = task.connect(params)

logger = task.get_logger()

# read dataset from the clearml
dataset = Dataset.get(
    dataset_project="NCI-ML-Project", dataset_name="Gcp_Cloud_Billing"
)
dataset_path = dataset.get_local_copy()
csv_path = os.path.join(dataset_path, "gcp_final_approved_dataset.csv")
df = pd.read_csv(csv_path)

# ✅ 特征工程逻辑（与你 preprocessing.py 保持一致）
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

# build pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            BaggingRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=params["regressor__estimator__max_depth"]
                ),
                n_estimators=params["regressor__n_estimators"],
                max_samples=params["regressor__max_samples"]
                / 100.0,  # 注意如果是百分数要转换
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

# # 先根据时间戳排序
# df_sorted = df.sort_values("Usage Start Date")

# # 重新提取X, y（保持和df_sorted一致的索引顺序）
# X_sorted = df_sorted[features]
# y_sorted = df_sorted[target]

# # 确定分割点（比如80%训练，20%测试）
# split_index = int(len(df_sorted) * 0.8)

# X_train = X_sorted.iloc[:split_index]
# X_test = X_sorted.iloc[split_index:]
# y_train = y_sorted.iloc[:split_index]
# y_test = y_sorted.iloc[split_index:]


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# measure of performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
logger.report_scalar("metrics", "RMSE", value=rmse, iteration=0)
logger.report_scalar("metrics", "R2", value=r2, iteration=1)
logger.report_scalar("metrics", "MAPE", value=mape, iteration=2)
print(f"✅ Done. RMSE: {rmse:.2f}, R2 Score: {r2:.4f}, MAPE: {mape:.2f}%")


plt.figure(figsize=(8, 4))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("Cloud Cost")
plt.legend()
plt.grid(True)

# upload figure to clearml
task.get_logger().report_matplotlib_figure(
    title="Prediction vs Actual (Line Plot)",
    series="Result Curve",
    figure=plt.gcf(),
    iteration=0,
)
plt.close()

dump(pipeline, filename="cost-model.joblib", compress=9)
