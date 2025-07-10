"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-29 18:49:57
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-29 18:49:59
FilePath: /cloud-cost-estimation/ml-models/preprocess.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

from typing import Any

import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(
        self, body: dict, state: dict, collect_custom_statistics_fn=None
    ) -> Any:
        # we expect to get two valid on the dict x0, and x1
        return [
            [body.get("x0", None), body.get("x1", None)],
        ]

    def postprocess(
        self, data: Any, state: dict, collect_custom_statistics_fn=None
    ) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=data.tolist() if isinstance(data, np.ndarray) else data)
