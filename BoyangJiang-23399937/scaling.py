"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-21 19:16:21
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-07-17 20:50:58
FilePath: /cloud-cost-estimation/BoyangJiang-23399937/scaling.py
Description:
scaling method comparison
Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import matplotlib.pyplot as plt

models = ["Original", "Standardized", "Normalized"]

# RMSE 和 R2 值
rmse_values = [2044.89, 1398.32, 1485.16]
r2_values = [0.2081, 0.6297, 0.5823]

fig, ax1 = plt.subplots(figsize=(8, 4))

color = "tab:blue"
ax1.set_xlabel("Scaling Method")
ax1.set_ylabel("RMSE", color=color)
ax1.plot(models, rmse_values, marker="o", label="RMSE", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim(0, max(rmse_values) + 500)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("R2 Score", color=color)
ax2.plot(models, r2_values, marker="s", label="R2", color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, 1)

plt.title("Comparison of Scaling Method Performance (RMSE vs R2)")
plt.grid(True)
plt.tight_layout()
plt.savefig("scaling_comparison.png")
