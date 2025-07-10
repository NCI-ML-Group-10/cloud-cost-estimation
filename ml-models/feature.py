"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-19 20:17:20
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-21 20:13:55
FilePath: /cloud-cost-estimation/ml-models/feature.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline, make_pipeline
from preprocessing import build_preprocessor, load_csv_data
import matplotlib.pyplot as plt

# 加载数据
df = load_csv_data()

# 构建预处理器、特征列和目标列
preprocessor, features, target = build_preprocessor()

X = df[features]
y = df[target]

# 划分训练集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 初始化 RFE + 随机森林（包裹法）
rfe = RFE(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    n_features_to_select=13,
    step=1,
)

# 构建 pipeline：预处理器 + 特征选择器
pipeline = make_pipeline(preprocessor, rfe)

# 拟合训练集
pipeline.fit(X_train, y_train)

# 获取经过预处理后的特征名
processed_feature_names = preprocessor.get_feature_names_out()

# 获取 RFE 的特征选择结果
selected_mask = rfe.support_
ranking = rfe.ranking_

# 构造结果 DataFrame
selected_result = pd.DataFrame(
    {
        "Feature": processed_feature_names,
        "Selected": selected_mask,
        "Ranking": ranking,
    }
).sort_values(by="Ranking")

print("💡 最优特征（Top 5）:")
print(selected_result[selected_result["Selected"] == True])
