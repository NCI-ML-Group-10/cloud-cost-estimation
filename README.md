<!--
 * @Author: Bryan x23399937@student.ncirl.ie
 * @Date: 2025-06-20 19:06:26
 * @LastEditors: Bryan x23399937@student.ncirl.ie
 * @LastEditTime: 2025-07-17 21:46:03
 * @FilePath: /cloud-cost-estimation/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by Bryan Jiang, All Rights Reserved. 
-->
# cloud-cost-estimation
A Machine Learning Framework For Predictive Cloud Cost Estimation in AIOps

## Dataset
- https://www.kaggle.com/datasets/sairamn19/gcp-cloud-billing-data


## Folder introduction
```
├── /.github/workflows/           # github actions cicd pipeline config file
├── /AshishChoudhary-23272805/    # team member source code
├── /BoyangJiang-23399937/        # team member source code
├── /MdAllYesasIslam-23413085/    # team member source code
├── /frontend-ui/                 # front code dir
├── /clearml-serving/             # clearml serving docker-compose file
├── /clearml-server/              # clearml server docker-compose file
```

## Cloud Cost Estimation web site

### frontend address: 
- http://cloud-cost-frontend.s3-website-us-east-1.amazonaws.com
### backend address: 
- http://clearml-serving.us-east-1.elasticbeanstalk.com:8080/serve/cloud_cost_predict

### Model Serving endpoint curl example

HTTP API:
curl -X POST "http://clearml-serving.us-east-1.elasticbeanstalk.com:8080/serve/cloud_cost_predict" -H "accept: application/json" -H "Content-Type: application/json" -d '
{
  "Service Name": "Cloud Storage",
  "Region/Zone": "us-west1",
  "Usage Quantity": 100.98,
  "CPU Utilization (%)": 0.65,
  "Memory Utilization (%)": 0.42,
  "Network Inbound Data (Bytes)": 102400,
  "Network Outbound Data (Bytes)": 204800
}'

## Publish a model step
### Training Your Model
```bash
python train.py
```

### Ensure Your Model is published status in clearml model registory

### Serving Register:

clearml-serving --id c0282de094604bc3aec22a6778473043 model add --engine sklearn --endpoint "cloud_cost_predict" --preprocess "inference_preprocess.py" --name "Bagging Tree Regressor Training - cost-model" --project "NCI-ML-Project"

#clearml-serving --id c0282de094604bc3aec22a6778473043 model auto-update --engine sklearn --endpoint "cloud_cost_predict" --preprocess "inference_preprocess.py" --name "Bagging Tree Regressor Model" --project "NCI-ML-Project"