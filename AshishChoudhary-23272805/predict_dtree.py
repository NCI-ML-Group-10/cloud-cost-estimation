import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load("model_output/tree_model.joblib")

# Simulate one input for prediction
input_data = pd.DataFrame([{
    'CPU Utilization (%)': 55,
    'Memory Utilization (%)': 70,
    'Network In (MB)': 200,
    'Network Out (MB)': 150,
    'Service Name': 'Compute Engine',
    'Region/Zone': 'us-central1'
}])

# Fit encoder on known values
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(pd.DataFrame([
    {'Service Name': 'Compute Engine', 'Region/Zone': 'us-central1'},
    {'Service Name': 'Cloud Storage', 'Region/Zone': 'us-west1'}
]))

# Encode input
encoded = pd.DataFrame(encoder.transform(input_data[['Service Name', 'Region/Zone']]),
                       columns=encoder.get_feature_names_out())

input_data = input_data.drop(columns=['Service Name', 'Region/Zone']).reset_index(drop=True)
final_input = pd.concat([input_data, encoded.reset_index(drop=True)], axis=1)

# Predict
prediction = model.predict(final_input)
print(f"Predicted Cost (INR): ₹{prediction[0]:.2f}")
