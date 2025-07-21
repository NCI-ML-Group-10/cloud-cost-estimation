import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Optional: ClearML (only if you use it)
from clearml import Task

def train_model():
    # 🔹 ClearML tracking
    task = Task.init(
        project_name="GCP Cost Estimation",
        task_name="Train Decision Tree Model",
        task_type="training"
    )

    # Load dataset
    df = pd.read_csv("gcp_final_approved_dataset.csv")

    # Drop irrelevant columns
    df = df.drop(columns=[
        'Resource ID', 'Usage Unit', 'Usage Start Date', 'Usage End Date',
        'Unrounded Cost ($)', 'Rounded Cost ($)'
    ])

    # Define features and target
    target = 'Total Cost (INR)'
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical variables
    categorical_cols = ['Service Name', 'Region/Zone']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(
        encoder.fit_transform(X[categorical_cols]),
        columns=encoder.get_feature_names_out()
    )

    # Merge numeric + encoded
    X_numeric = X.drop(columns=categorical_cols).reset_index(drop=True)
    X_encoded = X_encoded.reset_index(drop=True)
    X_final = pd.concat([X_numeric, X_encoded], axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # Train Decision Tree
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Save model and encoder
    os.makedirs("model_output", exist_ok=True)
    joblib.dump(model, "model_output/tree_model.joblib")
    joblib.dump(encoder, "model_output/encoder.joblib")

    print("✅ Model and encoder saved to model_output/")

if __name__ == "__main__":
    train_model()
