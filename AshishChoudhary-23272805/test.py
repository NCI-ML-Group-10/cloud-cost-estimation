import joblib

encoder = joblib.load("model_output/encoder.joblib")

print("✅ Accepted values for 'Service Name':")
print(encoder.categories_[0])

print("\n✅ Accepted values for 'Region/Zone':")
print(encoder.categories_[1])
