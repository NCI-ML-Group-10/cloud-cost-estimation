import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

from preprocess_dtree import clean_data
from feature_dtree import extract_features
from train_dtree import train_model  # External training function

CSV_PATH = "gcp_final_approved_dataset.csv"
MODEL_PATH = "model_output/tree_model.joblib"
ENCODER_PATH = "model_output/encoder.joblib"

def predict_from_csv_random(num_rows=3):
    print("🔍 Loading model for prediction...")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("🔁 Reading from CSV for prediction...")
    df = pd.read_csv(CSV_PATH)
    df = clean_data(df)

    input_data = df.sample(num_rows, random_state=np.random.randint(1000)).copy()
    actual_costs = input_data['Total Cost (INR)']
    input_features = input_data.drop(columns=['Total Cost (INR)'])

    # Encode categorical columns
    encoded = pd.DataFrame(encoder.transform(input_features[['Service Name', 'Region/Zone']]),
                           columns=encoder.get_feature_names_out())

    input_numeric = input_features.drop(columns=['Service Name', 'Region/Zone']).reset_index(drop=True)
    final_input = pd.concat([input_numeric, encoded.reset_index(drop=True)], axis=1)

    # Align with model's training features
    model_features = model.feature_names_in_
    final_input = final_input.reindex(columns=model_features, fill_value=0)

    # Make predictions
    predictions = model.predict(final_input)

    # Display results
    print(f"\n🧾 Showing predictions for {num_rows} random samples:\n")
    for i in range(num_rows):
        row = input_features.iloc[i]
        actual = actual_costs.iloc[i]
        predicted = predictions[i]

        # Accuracy metrics
        mae = abs(predicted - actual)
        rmse = np.sqrt((predicted - actual) ** 2)
        mape = (mae / actual) * 100 if actual != 0 else 0

        print(f"📦 Sample {i + 1}")
        print(f"Service Name                  : {row['Service Name']}")
        print(f"Usage Quantity                : {row['Usage Quantity']}")
        print(f"Region/Zone                   : {row['Region/Zone']}")
        print(f"CPU Utilization (%)           : {row['CPU Utilization (%)']}")
        print(f"Memory Utilization (%)        : {row['Memory Utilization (%)']}")
        print(f"Network Inbound Data (Bytes)  : {row['Network Inbound Data (Bytes)']}")
        print(f"Network Outbound Data (Bytes) : {row['Network Outbound Data (Bytes)']}")
        print(f"Cost per Quantity ($)         : {row['Cost per Quantity ($)']}")
        print(f"🔮 Predicted Cost (INR)        : ₹{predicted:,.2f}")
        print(f"📊 Actual Cost (INR)           : ₹{actual:,.2f}")
        accuracy_pct = 100 - mape  # Because accuracy = 100% - MAPE%

        print(f"🎯 Accuracy Metrics:")
        print(f"   🔹 MAE     : ₹{mae:,.2f}")
        print(f"   🔹 RMSE    : ₹{rmse:,.2f}")
        print(f"   🔹 MAPE    : {mape:.2f}%")
        print(f"   ✅ Accuracy: {accuracy_pct:.2f}%\n")




if __name__ == "__main__":
    print("🔧 Decision Tree Project CLI")
    print("[1] Train Model")
    print("[2] Predict from Random Dataset Rows")
    choice = input("Select option (1/2): ").strip()

    if choice == '1':
        train_model()
    elif choice == '2':
        predict_from_csv_random(num_rows=2)
    else:
        print("❌ Invalid selection.")

