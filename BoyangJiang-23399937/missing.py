"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-21 18:17:09
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-21 18:17:35
FilePath: /cloud-cost-estimation/ml-models/missing.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("~/Downloads/gcp_final_approved_dataset.csv")

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)
print("Total Missing Values:", missing_values.sum())
