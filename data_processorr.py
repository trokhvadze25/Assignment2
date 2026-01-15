import ipaddress
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("Darknet.csv")

# Drop non-useful columns
df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)

# Drop missing values
df = df.dropna()

# Convert IP addresses to integers (IPv4 only)
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return np.nan

df['Src IP'] = df['Src IP'].apply(ip_to_int)
df['Dst IP'] = df['Dst IP'].apply(ip_to_int)

# Drop rows where IP conversion failed
df = df.dropna()

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Encode labels
label_encoder = LabelEncoder()
df['Label1'] = label_encoder.fit_transform(df['Label1'])

# Separate features and label
X = df.drop('Label1', axis=1)
y = df['Label1']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

# Align labels with cleaned features
y = y.loc[X.index]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed dataset
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['Label1'] = y.values

processed_df.to_csv("processed.csv", index=False)

print("Preprocessing completed successfully.")
print(f"Final dataset shape: {processed_df.shape}")