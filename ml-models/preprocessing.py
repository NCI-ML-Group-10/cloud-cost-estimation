"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-19 22:18:50
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-20 19:24:05
FilePath: /cloud-cost-estimation/ml-models/preprocessing.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

# preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def build_preprocessor() -> tuple[ColumnTransformer, tuple, str]:
    onehot_cols = ["Service Name", "Region/Zone"]
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
        ]
    )

    return preprocessor, onehot_cols + num_cols, target


def load_csv_data() -> pd.DataFrame:
    # read data frame from csv
    df = pd.read_csv("~/Downloads/gcp_final_approved_dataset.csv")

    df["Service_Avg_Cost"] = df.groupby("Service Name")["Total Cost (INR)"].transform(
        "mean"
    )
    df["Service_Avg_Cost"] = df["Service_Avg_Cost"].fillna(0)

    df["Region_Zone_Avg_Cost"] = df.groupby("Region/Zone")[
        "Total Cost (INR)"
    ].transform("mean")
    df["Region_Zone_Avg_Cost"] = df["Region_Zone_Avg_Cost"].fillna(0)

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
    df["is_night_usage"] = df["start_hour"].apply(
        lambda h: 1 if h <= 6 or h >= 22 else 0
    )

    # total duraing time
    df["duration_minutes"] = (
        df["Usage End Date"] - df["Usage Start Date"]
    ).dt.total_seconds() / 60
    df["duration_minutes"] = df["duration_minutes"].fillna(0)
    return df
