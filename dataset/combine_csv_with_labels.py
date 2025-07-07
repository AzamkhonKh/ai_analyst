import pandas as pd
import os

# Paths to the CSV files
broken_path = os.path.join('BrokenTooth', 'b30hz90.csv')
healthy_path = os.path.join('Healthy', 'h30hz90.csv')

# Load the CSVs
df_broken = pd.read_csv(broken_path)
df_healthy = pd.read_csv(healthy_path)

# Add label columns
df_broken['label'] = 'KO'
df_healthy['label'] = 'OK'

# Combine the dataframes
df_combined = pd.concat([df_broken, df_healthy], ignore_index=True)

# Save the combined dataframe
output_path = 'combined_labeled.csv'
df_combined.to_csv(output_path, index=False)

print(f"Combined CSV saved to {output_path} with labels.") 