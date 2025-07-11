import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from clearml import Task, Dataset, Model, OutputModel

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

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

# -- K折交叉验证 --
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折

# RMSE需要自定义scorer
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False,
)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# 计算每折分数（注意：sklearn对rmse/mae输出负数，需取反）
r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
rmse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring=rmse_scorer)
mae_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring=mae_scorer)
logger.report_scalar("metrics", "RMSE", value=rmse_scores.mean(), iteration=0)
logger.report_scalar("metrics", "R2", value=r2_scores.mean(), iteration=1)
logger.report_scalar("metrics", "MAPE", value=mae_scores.mean(), iteration=2)
print(f"5-Fold CV R²:   Mean={r2_scores.mean():.4f}, Std={r2_scores.std():.4f}")
print(f"5-Fold CV RMSE: Mean={rmse_scores.mean():.2f}, Std={rmse_scores.std():.2f}")
print(f"5-Fold CV MAE:  Mean={mae_scores.mean():.2f}, Std={mae_scores.std():.2f}")

# 可以将每折结果画成箱线图/柱状图
plt.figure(figsize=(7, 5))
plt.boxplot([rmse_scores, mae_scores, r2_scores], tick_labels=["RMSE", "MAE", "R²"])
plt.title("5-Fold Cross-Validation Results")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()

task.get_logger().report_matplotlib_figure(
    title="5-Fold Cross-Validation Results",
    series="CV Boxplot",
    figure=plt.gcf(),
    iteration=99,
)
# plt.show()
plt.close()

dump(pipeline, filename="cost-model.joblib", compress=9)

for i in task.get_models().output:
    print(i.id)
    Model(model_id=i.id).publish()
