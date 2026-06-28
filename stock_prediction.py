import pickle
import pandas as pd

with open("stock_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

new_data = pd.DataFrame({
    "Open":[420],
    "High":[425],
    "Low":[418],
    "Volume":[50000000],
    "Daily_Range":[7],
    "MA_10":[419],
    "MA_50":[405]
})

new_scaled = loaded_scaler.transform(new_data)

prediction = loaded_model.predict(new_scaled)

print(prediction)