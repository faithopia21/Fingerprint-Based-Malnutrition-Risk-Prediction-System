import pandas as pd
from pathlib import Path

# Paths to XPT files
bmx_file = Path("BMX_J.XPT")
demo_file = Path("DEMO_J.XPT")

# Load datasets
bmx_df = pd.read_sas(bmx_file, format="xport")
demo_df = pd.read_sas(demo_file, format="xport")

# Merge on SEQN
nhanes_df = pd.merge(demo_df, bmx_df, on="SEQN", how="inner")

# Columns we want (but only keep those that exist in dataset)
columns_to_keep = [
    "SEQN",       # ID
    "RIAGENDR",   # Gender
    "RIDAGEYR",   # Age in years
    "BMXWT",      # Weight (kg)
    "BMXHT",      # Height (cm)
    "BMXBMI",     # BMI
    "BMXWAIST",   # Waist circumference (cm)
    "BMXHIP",     # Hip circumference (cm)
    "BMXCALF",    # Calf circumference (cm) – may not exist in all years
]

# Keep only the columns that exist
columns_available = [col for col in columns_to_keep if col in nhanes_df.columns]
nhanes_df = nhanes_df[columns_available]

# Save cleaned dataset
nhanes_df.to_csv("NHANES_master_dataset.csv", index=False)

print("NHANES_master_dataset.csv created with columns:", list(nhanes_df.columns))