import pandas as pd
import requests

# Load the dataset
df = pd.read_csv("gcp_final_approved_dataset.csv")

# Select a random row with all required columns
sample = df[[  
    "Service Name", "Region/Zone", "Usage Quantity",
    "CPU Utilization (%)", "Memory Utilization (%)",
    "Network Inbound Data (Bytes)", "Network Outbound Data (Bytes)",
    "Cost per Quantity ($)", "Total Cost (INR)"
]].sample(1).reset_index(drop=True)

# Prepare payload for FastAPI
data = {
    "Service_Name": sample["Service Name"][0],
    "Region_Zone": sample["Region/Zone"][0],
    "Usage_Quantity": float(sample["Usage Quantity"][0]),
    "CPU_Utilization": float(sample["CPU Utilization (%)"][0]),
    "Memory_Utilization": float(sample["Memory Utilization (%)"][0]),
    "Network_Inbound_Data": float(sample["Network Inbound Data (Bytes)"][0]),
    "Network_Outbound_Data": float(sample["Network Outbound Data (Bytes)"][0]),
    "Cost_Per_Quantity": float(sample["Cost per Quantity ($)"][0]),  
    "Actual_Cost_INR": float(sample["Total Cost (INR)"][0])
}

# Send request to FastAPI
url = "http://localhost:8000/predict"
response = requests.post(url, json=data)
result = response.json()

# Display inputs
print(" Sample Input Data:")
print(f"  • Service Name              : {data['Service_Name']}")
print(f"  • Region/Zone               : {data['Region_Zone']}")
print(f"  • Usage Quantity            : {data['Usage_Quantity']}")
print(f"  • CPU Utilization (%)       : {data['CPU_Utilization']}")
print(f"  • Memory Utilization (%)    : {data['Memory_Utilization']}")
print(f"  • Network Inbound Data      : {data['Network_Inbound_Data']}")
print(f"  • Network Outbound Data     : {data['Network_Outbound_Data']}")
print(f"  • Cost per Quantity ($)     : {data['Cost_Per_Quantity']}")
print(f"  • Actual Cost (INR)         : ₹{data['Actual_Cost_INR']:,.2f}\n")

# Display output
print(" API Prediction & Metrics:")
print(f"  • Predicted Cost (INR)      : {result['Predicted Cost (INR)']}")
print(f"  • MAE                       : {result['MAE']}")
print(f"  • RMSE                      : {result['RMSE']}")
print(f"  • MAPE                      : {result['MAPE']}")
print(f"  • Accuracy                  : {result['Accuracy']}")
