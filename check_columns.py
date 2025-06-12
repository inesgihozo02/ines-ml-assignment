import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Print the column names
print("Columns in your dataset:")
print(df.columns.tolist())

