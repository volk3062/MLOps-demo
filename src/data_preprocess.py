import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop irrelevant columns
df.drop(['customerID'], axis=1, inplace=True)

# Convert categorical to numeric
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

df.to_csv("data/processed/cleaned.csv", index=False)
print("✅ Data cleaned and saved.")
