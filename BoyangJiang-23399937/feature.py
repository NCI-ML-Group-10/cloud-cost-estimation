"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-19 20:17:20
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-07-17 20:56:01
FilePath: /cloud-cost-estimation/BoyangJiang-23399937/feature.py
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

df = load_csv_data()

preprocessor, features, target = build_preprocessor()

X = df[features]
y = df[target]

# split data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# init
rfe = RFE(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    n_features_to_select=13,
    step=1,
)

# make pipeline
pipeline = make_pipeline(preprocessor, rfe)

# fitting training dataset
pipeline.fit(X_train, y_train)

# get processed feature name
processed_feature_names = preprocessor.get_feature_names_out()

# get rfe feature selection result
selected_mask = rfe.support_
ranking = rfe.ranking_

# make result DataFrame
selected_result = pd.DataFrame(
    {
        "Feature": processed_feature_names,
        "Selected": selected_mask,
        "Ranking": ranking,
    }
).sort_values(by="Ranking")

print("💡 best features（Top 5）:")
print(selected_result[selected_result["Selected"] == True])
