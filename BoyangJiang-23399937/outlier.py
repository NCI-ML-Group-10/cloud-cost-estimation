"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-21 17:29:02
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-07-17 20:47:08
FilePath: /cloud-cost-estimation/BoyangJiang-23399937/outlier.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# read data set
file_path = "~/Downloads/gcp_final_approved_dataset.csv"
df = pd.read_csv(file_path)

# choose specific fields
selected_columns = [
    "Usage Quantity",
    "CPU Utilization (%)",
    "Memory Utilization (%)",
    "Network Inbound Data (Bytes)",
    "Network Outbound Data (Bytes)",
    "Cost per Quantity ($)",
]
df_selected = df[selected_columns]
df.replace(["?", -1, -99, -999], np.nan, inplace=True)

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

outliers = df_selected[labels == -1]

# pca visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[labels != -1, 0], X_pca[labels != -1, 1], c="blue", label="Normal")
plt.scatter(X_pca[labels == -1, 0], X_pca[labels == -1, 1], c="red", label="Outliers")
plt.title("DBSCAN Outlier Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outlier_detection_plot.png")
