import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# 读取数据

file_path = "~/Downloads/gcp_final_approved_dataset.csv"
df = pd.read_csv(file_path)

# 选取指定字段
selected_columns = [
    "Usage Quantity",
    "CPU Utilization (%)",
    "Memory Utilization (%)",
    "Network Inbound Data (Bytes)",
    "Network Outbound Data (Bytes)",
]
df_selected = df[selected_columns]
df.replace(["?", -1, -99, -999], np.nan, inplace=True)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 异常值索引
outliers = df_selected[labels == -1]

# 降维可视化
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
