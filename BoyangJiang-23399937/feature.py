import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("~/Downloads/gcp_final_approved_dataset.csv")

# define target field
target = "Rounded Cost ($)"
features = [col for col in df.columns if col != target]

# automatic labelencode all the field that are not numberic
for col in features:
    if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
        df[col] = df[col].astype(str).fillna("NAN")
        df[col] = LabelEncoder().fit_transform(df[col])

X = df[features]
y = df[target]

model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# fields importance
importances = model.feature_importances_
feat_names = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(14, 6))
plt.bar([feat_names[i] for i in indices], importances[indices])
plt.yscale("log")  # 对数刻度
plt.xlabel("Feature Selection")
plt.ylabel("Importance (log scale)")
plt.title("Feature Importance for All Raw Fields (No OneHot)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
# plt.show()
plt.savefig("feature_importance.png")

# 如需查看数值
for i in indices:
    print(f"{feat_names[i]}: {importances[i]}")
