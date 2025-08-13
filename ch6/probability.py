import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Configurations
start_date = datetime(2030, 7, 1)
days = 31  
locations = ['New York', 'Chicago', 'Los Angeles']

# Generate Data 
data = []
for location in locations:
    for day in range(days):
        date = start_date + timedelta(days=day)

        # Predicted probability of rain
        predicted_prob = np.round(np.random.uniform(0, 1), 2)

        # Actual result based on the predicted probability
        actual_rain = np.random.rand() < predicted_prob
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Location': location,
            'Predicted_Rain_Prob': predicted_prob,
            'Actual_Rain': int(actual_rain)
        })

df = pd.DataFrame(data)

print(df.head())

