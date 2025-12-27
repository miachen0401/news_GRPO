"""Take 50 random rows for training and 10 different rows for validation"""

import pandas as pd

# Read the train parquet file
df = pd.read_parquet("data/gsm8k/train.parquet")

# Sample 60 random rows total
sample_df = df.sample(n=60, random_state=42)

# Split: first 50 for training, last 10 for validation
train_df = sample_df.iloc[:50]
val_df = sample_df.iloc[50:]

# Save to files
train_df.to_parquet("data/gsm8k/train_sample.parquet", index=False)
val_df.to_parquet("data/gsm8k/val_sample.parquet", index=False)

print(f"Original rows: {len(df)}")
print(f"Training samples: {len(train_df)} → train_sample.parquet")
print(f"Validation samples: {len(val_df)} → val_sample.parquet")
