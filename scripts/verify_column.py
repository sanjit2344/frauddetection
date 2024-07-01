import pandas as pd

# Load the new data
new_data_path = 'data/external/alpha_vantage_data.csv'
df = pd.read_csv(new_data_path)

# Verify the columns of the new data
print("Columns in new data:", df.columns.tolist())
