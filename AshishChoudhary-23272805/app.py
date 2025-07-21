from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess_dtree import clean_data
import numpy as np


#  Define input schema
class InputData(BaseModel):
    Service_Name: str
    Region_Zone: str
    Usage_Quantity: float
    CPU_Utilization: float
    Memory_Utilization: float
    Network_Inbound_Data: float
    Network_Outbound_Data: float
    Cost_Per_Quantity: float
    Actual_Cost_INR: float  # Only for metrics

#  Initialize FastAPI
app = FastAPI()

#  Load model and encoder once
model = joblib.load("model_output/tree_model.joblib")
encoder = joblib.load("model_output/encoder.joblib")

@app.get("/")
def root():
    return {"message": " GCP Cost Prediction API is running!"}

#  POST prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    # Convert to dict and pull actual cost
    data_dict = input_data.dict()
    actual_cost = data_dict.pop("Actual_Cost_INR")

    #  Create DataFrame & rename columns
    df = pd.DataFrame([data_dict])
    df = df.rename(columns={
        "Service_Name": "Service Name",
        "Region_Zone": "Region/Zone",
        "Usage_Quantity": "Usage Quantity",
        "CPU_Utilization": "CPU Utilization (%)",
        "Memory_Utilization": "Memory Utilization (%)",
        "Network_Inbound_Data": "Network Inbound Data (Bytes)",
        "Network_Outbound_Data": "Network Outbound Data (Bytes)",
        "Cost_Per_Quantity": "Cost per Quantity ($)"
    })

    #  Apply same preprocessing as local prediction
    # df = clean_data(df)

    #  Encode categorical features
    encoded = pd.DataFrame(
        encoder.transform(df[["Service Name", "Region/Zone"]]),
        columns=encoder.get_feature_names_out()
    )

    #  Numeric features
    numeric_df = df.drop(columns=["Service Name", "Region/Zone"]).reset_index(drop=True)
    final_input = pd.concat([numeric_df, encoded.reset_index(drop=True)], axis=1)

    #  Ensure column order matches model
    final_input = final_input.reindex(columns=model.feature_names_in_, fill_value=0)

    #  Predict
    predicted_cost = model.predict(final_input)[0]

    # Metrics
    mae = mean_absolute_error([actual_cost], [predicted_cost])
    rmse = np.sqrt(mean_squared_error([actual_cost], [predicted_cost]))
    mape = abs((actual_cost - predicted_cost) / actual_cost) * 100
    accuracy = 100 - mape

    #  Clean output
    return {
        "Predicted Cost (INR)": f"₹{predicted_cost:,.2f}",
        "Actual Cost (INR)": f"₹{actual_cost:,.2f}",
        "MAE": f"₹{mae:,.2f}",
        "RMSE": f"₹{rmse:,.2f}",
        "MAPE": f"{mape:.2f}%",
        "Accuracy": f"{accuracy:.2f}%"
    }
