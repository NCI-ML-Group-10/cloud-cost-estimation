"""
Author: Bryan x23399937@student.ncirl.ie
Date: 2025-06-28 21:00:38
LastEditors: Bryan x23399937@student.ncirl.ie
LastEditTime: 2025-07-17 20:38:42
FilePath: /cloud-cost-estimation/BoyangJiang-23399937/hpo.py
Description:

Copyright (c) 2025 by Bryan Jiang, All Rights Reserved.
"""

from clearml import Task
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

# init the HPO task
task = Task.init(
    project_name="NCI-ML-Project",
    task_name="Bagging Tree HPO",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

base_task_id = "7fc64128b0914eb691e527a6e15712fc"

# define the high parameter scope
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformIntegerParameterRange(
            "regressor__n_estimators", min_value=10, max_value=300, step_size=10
        ),
        UniformIntegerParameterRange(
            "regressor__estimator__max_depth", min_value=2, max_value=10, step_size=1
        ),
    ],
    objective_metric_title="metrics",
    objective_metric_series="RMSE",
    objective_metric_sign="min",
    optimizer_class=OptimizerOptuna,
    execution_queue="default",
    max_number_of_concurrent_tasks=4,
    optimization_time_limit=60.0,
    compute_time_limit=120,
    total_max_jobs=30,
    min_iteration_per_job=1,
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


# start optimizer
optimizer.set_time_limit(in_minutes=6)
optimizer.start_locally(job_complete_callback=job_complete_callback)

# wait for all the tasks finish
optimizer.wait()

optimizer.stop()
print("HPO task is done!")
