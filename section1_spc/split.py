import pandas as pd
from sklearn.model_selection import train_test_split

#Load data
df = pd.read_csv("dragon_data.csv")
print(f"Loaded {len(df)} rows.")
print("Species counts (raw):")
print(df["SPC"].value_counts())

#typo fix
df["SPC"] = df["SPC"].replace("Wyvernn", "Wyvern")
print("\nSpecies counts (after cleaning):")
print(df["SPC"].value_counts())

# One-hot encode SPC (drop "Wyvern" as reference category)
dummies = pd.get_dummies(df["SPC"], prefix="SPC", drop_first=False)
# drop_first=False gives us all three, manually drop Wyvern
dummies = dummies.drop(columns=["SPC_Wyvern"])

#Convert booleans to integers (0/1) for Excel readability
dummies = dummies.astype(int)

#Combine back with original data
df_encoded = pd.concat([df, dummies], axis=1)

# Columns we want in the output:
# SPC (original), SPC_Dragon, SPC_Hydra, FHO
output_cols = ["SPC", "SPC_Dragon", "SPC_Hydra", "FHO"]
df_model = df_encoded[output_cols].copy()

print("\nSample of encoded data (first 5 rows):")
print(df_model.head())

# Train Test split  (80% train, 20% test)
train_df, test_df = train_test_split(df_model, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"\nTrain size : {len(train_df)} rows")
print(f"Test size  : {len(test_df)} rows")

train_df.to_csv("spc_train.csv", index=False)
test_df.to_csv("spc_test.csv",  index=False)

df_model.to_csv("spc_all.csv", index=False)

