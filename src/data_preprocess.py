import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob
import os

# Define paths for clarity
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_FILE = "data/processed/cleaned.csv"

# Use glob to find all .csv files in the raw data directory
all_csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

if not all_csv_files:
    print("No CSV files found in data/raw. Exiting.")
    # We exit gracefully if there's nothing to process
    exit()

# Read all found CSVs and combine them into a single DataFrame
df_list = [pd.read_csv(file) for file in all_csv_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Loaded and combined {len(all_csv_files)} file(s) from {RAW_DATA_DIR}.")

# --- Your Processing Logic ---
# Drop irrelevant columns if it exists
if "customerID" in df.columns:
    df.drop(["customerID"], axis=1, inplace=True)

# Convert all object columns to numeric categories
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype('category').cat.codes

# Remove any duplicate rows that might result from combining files
df.drop_duplicates(inplace=True)

# Ensure the processed directory exists
os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)

# Overwrite the cleaned.csv file with the newly processed data
df.to_csv(PROCESSED_DATA_FILE, index=False)

print(f"✅ Data processing complete. Cleaned data saved to {PROCESSED_DATA_FILE}.")




# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# # Drop irrelevant columns
# df.drop(["customerID"], axis=1, inplace=True)

# # Convert categorical to numeric
# for col in df.select_dtypes(include=["object"]).columns:
#     df[col] = LabelEncoder().fit_transform(df[col])

# df.to_csv("data/processed/cleaned.csv", index=False)
# print("✅ Data cleaned and saved.")
