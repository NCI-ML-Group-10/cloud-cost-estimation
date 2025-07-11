<!--
 * @Author: Bryan x23399937@student.ncirl.ie
 * @Date: 2025-06-20 19:06:26
 * @LastEditors: Bryan x23399937@student.ncirl.ie
 * @LastEditTime: 2025-07-11 11:59:29
 * @FilePath: /cloud-cost-estimation/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by Bryan Jiang, All Rights Reserved. 
-->
# cloud-cost-estimation




HTTP API:
curl -X POST "http://clearml-serving.us-east-1.elasticbeanstalk.com:8080/serve/cloud_cost_predict/1" -H "accept: application/json" -H "Content-Type: application/json" -d '
{
  "Service Name": "Cloud Storage",
  "Region/Zone": "us-west1",
  "Usage Quantity": 100.98,
  "CPU Utilization (%)": 0.65,
  "Memory Utilization (%)": 0.42,
  "Network Inbound Data (Bytes)": 102400,
  "Network Outbound Data (Bytes)": 204800
}'


serving register:

clearml-serving --id c0282de094604bc3aec22a6778473043 model auto-update --engine sklearn --endpoint "cloud_cost_predict" --preprocess "inference_preprocess.py" --name "Bagging Tree Regressor Model" --project "NCI-ML-Project"


clearml-serving --id c0282de094604bc3aec22a6778473043 model add --engine sklearn --endpoint "cloud_cost_predict" --preprocess "inference_preprocess.py" --name "Bagging Tree Regressor Training - cost-model" --project "NCI-ML-Project"