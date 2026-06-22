import pandas as pd

df = pd.read_csv("cleaned_creditcard.csv")   # Use the cleaned file

print(df.shape)

print(df["Class"].isnull().sum())      # NaNs in target
print(df.isnull().sum().sum())         # Total NaNs
