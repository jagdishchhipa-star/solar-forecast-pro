import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Create realistic training data (1kW peak system)
np.random.seed(42)
n_hours = 20000 

ghi = np.random.uniform(0, 1000, n_hours) # Sun intensity
temp = np.random.uniform(5, 45, n_hours)   # Air temperature

# Solar Physics Equation: Power = (Sun/1000) * Capacity * Efficiency_Loss
# Loss: -0.4% for every degree above 25°C
power = (ghi / 1000) * 1.0 * (1 + (-0.004 * (temp - 25)))
power = np.clip(power, 0, None) * 0.95 # 5% system/cable loss

# 2. Train the AI
df = pd.DataFrame({'ghi': ghi, 'temp': temp, 'power': power})
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[['ghi', 'temp']], df['power'])

# 3. Save it
joblib.dump(model, 'solar_model.pkl')
print("✅ Brain (solar_model.pkl) created!")