"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-28 21:00:38
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-06-29 XX:XX:XX
FilePath: /cloud-cost-estimation/ml-models/optima.py
Description: ClearML HPO script for RandomForest pipeline

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

from clearml import Task
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

# ✅ 初始化优化任务
task = Task.init(
    project_name="NCI-ML-Project",
    task_name="Bagging Tree HPO",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

# ✅ 传入你要优化的 base 训练任务 ID（来自 UI 中 "cost_training" 的 ID）
# 例如：你训练脚本中通过 RandomForestRegressor 构建的 pipeline 任务
# 在 Web UI 中复制它的任务 ID
base_task_id = "7fc64128b0914eb691e527a6e15712fc"  # ← 请替换成实际 ID

# ✅ 配置超参数范围
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformIntegerParameterRange(
            "regressor__n_estimators", min_value=10, max_value=300, step_size=10
        ),
        # UniformIntegerParameterRange(
        #     "regressor__max_samples", min_value=50, max_value=100, step_size=10
        # ),
        UniformIntegerParameterRange(
            "regressor__estimator__max_depth", min_value=2, max_value=10, step_size=1
        ),
    ],
    objective_metric_title="metrics",  # ✅ 与训练中 report_scalar 的 title 保持一致
    objective_metric_series="RMSE",  # ✅ 与 report_scalar 的 series 保持一致
    objective_metric_sign="min",  # ✅ 误差越小越好
    optimizer_class=OptimizerOptuna,  # ✅ 使用 Optuna 算法
    execution_queue="default",  # ✅ 默认任务执行队列
    max_number_of_concurrent_tasks=4,
    optimization_time_limit=60.0,  # 最长优化时间（分钟）
    compute_time_limit=120,  # 单个任务最大计算时间（分钟）
    total_max_jobs=30,  # 最多尝试 20 个组合
    min_iteration_per_job=1,  # sklearn 无迭代，设为1即可
    max_iteration_per_job=1,
)


def job_complete_callback(
    job_id,  # type: str
    objective_value,  # type: float
    objective_iteration,  # type: int
    job_parameters,  # type: dict
    top_performance_job_id,  # type: str
):
    print(
        "Job completed!", job_id, objective_value, objective_iteration, job_parameters
    )
    if job_id == top_performance_job_id:
        print(
            "WOOT WOOT we broke the record! Objective reached {}".format(
                objective_value
            )
        )


# ✅ 启动优化器
optimizer.set_time_limit(in_minutes=6)
optimizer.start_locally(job_complete_callback=job_complete_callback)

# ✅ 等待所有任务完成
optimizer.wait()

optimizer.stop()
print("✅ 超参数优化已完成")
