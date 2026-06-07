import os
import pandas as pd

# Set your folder path (update this to where the WHO and NHANES files are stored)
DATA_DIR = r"c:\Users\DELL\OneDrive\Documents\Portfolio\Projects\LAUTECH\Malnutrition Risk Prediction System"

# WHO growth standards mapping
files = {
    "wfa_boys":   ("wfa_boys_0-to-5-years_zscores.xlsx", "WFA", "M", "0-5y"),
    "wfa_girls":  ("wfa_girls_0-to-5-years_zscores.xlsx", "WFA", "F", "0-5y"),

    "lhfa_boys_0_2": ("lhfa_boys_0-to-2-years_zscores.xlsx", "HFA", "M", "0-2y"),
    "lhfa_boys_2_5": ("lhfa_boys_2-to-5-years_zscores.xlsx", "HFA", "M", "2-5y"),
    "lhfa_girls_0_2":("lhfa_girls_0-to-2-years_zscores.xlsx", "HFA", "F", "0-2y"),
    "lhfa_girls_2_5":("lhfa_girls_2-to-5-years_zscores.xlsx", "HFA", "F", "2-5y"),

    "wfh_boys":   ("wfh_boys_2-to-5-years_zscores.xlsx", "WFH", "M", "2-5y"),
    "wfh_girls":  ("wfh_girls_2-to-5-years_zscores.xlsx", "WFH", "F", "2-5y"),

    "wfl_boys":   ("wfl_boys_0-to-2-years_zscores.xlsx", "WFL", "M", "0-2y"),
    "wfl_girls":  ("wfl_girls_0-to-2-years_zscores.xlsx", "WFL", "F", "0-2y"),
}

# Function to load, standardize, and tag WHO dataset
def load_and_tag(filename, indicator, sex, age_range):
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(filepath, engine="openpyxl")

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Rename common columns
    rename_map = {
        "Age": "AgeMonths",
        "Age_month": "AgeMonths",
        "Month": "AgeMonths",
        "Height": "HeightCM",
        "Length": "LengthCM",
        "Weight": "WeightKG",
        "SD": "SD_Value",
        "Z": "Z_Score",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # Add metadata
    df["Indicator"] = indicator
    df["Sex"] = sex
    df["AgeRange"] = age_range
    df["Source"] = "WHO"

    return df

# --- Load WHO datasets ---
who_datasets = []
for key, (filename, indicator, sex, age_range) in files.items():
    df = load_and_tag(filename, indicator, sex, age_range)
    who_datasets.append(df)

who_master_df = pd.concat(who_datasets, ignore_index=True)

# --- Load NHANES dataset ---
nhanes_file = os.path.join(DATA_DIR, "NHANES_master_dataset.csv")  # update file name if needed
nhanes_df = pd.read_csv(nhanes_file)

# Standardize NHANES column names to match WHO where possible
nhanes_df = nhanes_df.rename(
    columns={
        "Gender": "Sex",     # assuming NHANES has Gender column
        "Age": "AgeMonths",  # adjust if in years
        "Weight": "WeightKG",
        "Height": "HeightCM",
        "Length": "LengthCM"
    }
)
nhanes_df["Source"] = "NHANES"
nhanes_df["Indicator"] = None  # since NHANES is raw data, not z-scores
nhanes_df["AgeRange"] = None   # can be computed later if needed

# --- Combine WHO + NHANES ---
combined_df = pd.concat([who_master_df, nhanes_df], ignore_index=True)

# Save combined dataset
output_file = os.path.join(DATA_DIR, "who_master_dataset.csv")
combined_df.to_csv(output_file, index=False)

print(f"[OK] Combined WHO + NHANES dataset saved: {output_file}")
print(f"Rows: {len(combined_df)}, Columns: {combined_df.shape[1]}")
print(combined_df.head(10))