from typing import Any, Dict
import numpy as np
import datetime as dt
import pandas as pd


# ClearML-Serving require the class name must be Preprocess
class Preprocess(object):
    def __init__(self):
        print(
            "-----------------this is boyang jiang model serving--------444-----------"
        )
        pass

    def preprocess(
        self, body: Dict, state: dict, collect_custom_statistics_fn=None
    ) -> Any:
        print(
            "-------------------this is boyang jiang model preprocess----444------------"
        )
        """
        from HTTP JSON get request data
        """
        service_name = body.get("Service Name", "Unknown")
        region_zone = body.get("Region/Zone", "Unknown")
        usage_quantity = float(body.get("Usage Quantity", 0.0))
        cpu_util = float(body.get("CPU Utilization (%)", 0.0))
        mem_util = float(body.get("Memory Utilization (%)", 0.0))
        net_in = float(body.get("Network Inbound Data (Bytes)", 0.0))
        net_out = float(body.get("Network Outbound Data (Bytes)", 0.0))
        cost_per_quantity = float(body.get("Cost per Quantity ($)", 0.0))
        total_cost = float(body.get("Total Cost (INR)", 0.0))

        # time handling
        try:
            start_dt = dt.datetime.strptime(
                body.get("Usage Start Date", ""), "%d-%m-%Y %H:%M"
            )
            end_dt = dt.datetime.strptime(
                body.get("Usage End Date", ""), "%d-%m-%Y %H:%M"
            )
        except Exception:
            start_dt = end_dt = dt.datetime.now()

        start_hour = start_dt.hour
        start_day_of_week = start_dt.weekday()
        start_month = start_dt.month
        is_weekend = int(start_day_of_week in [5, 6])
        is_night_usage = int(start_hour <= 6 or start_hour >= 22)
        duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
        duration_minutes = max(duration_minutes, 0.0)

        service_avg_cost = total_cost
        region_zone_avg_cost = total_cost

        return pd.DataFrame(
            [
                [
                    usage_quantity,
                    cpu_util,
                    mem_util,
                    net_in,
                    net_out,
                    cost_per_quantity,
                    service_avg_cost,
                    region_zone_avg_cost,
                    start_hour,
                    start_day_of_week,
                    start_month,
                    is_weekend,
                    is_night_usage,
                    duration_minutes,
                    service_name,
                    region_zone,
                ]
            ],
            columns=[
                "Usage Quantity",
                "CPU Utilization (%)",
                "Memory Utilization (%)",
                "Network Inbound Data (Bytes)",
                "Network Outbound Data (Bytes)",
                "Cost per Quantity ($)",
                "Service_Avg_Cost",
                "Region_Zone_Avg_Cost",
                "start_hour",
                "start_day_of_week",
                "start_month",
                "is_weekend",
                "is_night_usage",
                "duration_minutes",
                "Service Name",
                "Region/Zone",
            ],
        )

    def postprocess(
        self, data: Any, state: dict, collect_custom_statistics_fn=None
    ) -> Dict:
        """
        return the result
        """
        return {
            "predicted_cost_usd": (
                data.tolist() if isinstance(data, np.ndarray) else data
            )
        }
