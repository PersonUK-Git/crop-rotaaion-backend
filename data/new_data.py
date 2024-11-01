import pandas as pd
import numpy as np

# Load the original data
file_path = 'Crop_recommendation.csv'
original_data = pd.read_csv(file_path)

# Define the number of synthetic rows needed
target_rows = 7000
num_original_rows = len(original_data)
num_synthetic_rows = target_rows - num_original_rows

# Define the range for small random adjustments per feature
adjustment_range = {
    'N': 5,  # Nitrogen
    'P': 5,  # Phosphorus
    'K': 5,  # Potassium
    'temperature': 1,  # Temperature (°C)
    'humidity': 2,     # Humidity (%)
    'ph': 0.2,         # pH level
    'rainfall': 5      # Rainfall (mm)
}

# Create synthetic data by making slight adjustments
synthetic_data = pd.DataFrame(columns=original_data.columns)

for _ in range(num_synthetic_rows):
    # Select a random row to base the synthetic data on
    row = original_data.sample(n=1).copy()

    # Adjust each numeric column by a random amount within the specified range
    for column, range_val in adjustment_range.items():
        if column in row.columns:
            # Apply a small random float adjustment within the range for finer control
            row[column] += np.random.uniform(-range_val, range_val)
    
    # Append the adjusted row to synthetic_data
    synthetic_data = pd.concat([synthetic_data, row], ignore_index=True)

# Combine the original and synthetic data
expanded_data = pd.concat([original_data, synthetic_data], ignore_index=True)

# Save the expanded dataset to a new CSV
expanded_file_path = 'Expanded_Crop_recommendation.csv'
expanded_data.to_csv(expanded_file_path, index=False)
expanded_file_path