import pandas as pd
import numpy as np
import os

# Paths to your datasets
DATA_DIR = r"c:\Users\DELL\OneDrive\Documents\Portfolio\Projects\LAUTECH\Malnutrition Risk Prediction System"
fingerprint_file = os.path.join(DATA_DIR, "fingerprint_features.csv")
nhanes_file = os.path.join(DATA_DIR, "NHANES_master_dataset.csv")
who_file = os.path.join(DATA_DIR, "who_master_dataset.csv")
output_file = os.path.join(DATA_DIR, "combined_fingerprint_nutrition.csv")

# Load datasets
print("Loading datasets...")
fingerprints = pd.read_csv(fingerprint_file)
nhanes = pd.read_csv(nhanes_file)
who = pd.read_csv(who_file)

print(f"Fingerprints loaded: {len(fingerprints)} records")
print(f"NHANES loaded: {len(nhanes)} records")
print(f"WHO loaded: {len(who)} records")

# Data validation and cleaning
print("\n=== Data Validation ===")
print("Fingerprint features summary:")
print(fingerprints[['ridge_density', 'minutiae_count']].describe())

# Check for any missing or invalid values in fingerprint features
print("\nMissing values in fingerprint features:")
print(fingerprints[['ridge_density', 'minutiae_count']].isnull().sum())

# Remove any rows with missing fingerprint features
fingerprints_clean = fingerprints.dropna(subset=['ridge_density', 'minutiae_count']).copy()
print(f"\nAfter cleaning: {len(fingerprints_clean)} valid fingerprint records")

# Step 1: Implement realistic age/sex assignment based on fingerprint characteristics
print("\nImplementing realistic age/sex assignment based on fingerprint characteristics...")

# Generate Subject_ID for fingerprints
fingerprints_clean['Subject_ID'] = ["Subject_{:03d}".format(i+1) for i in range(len(fingerprints_clean))]

# Normalize ridge density to 0-1 scale if not already normalized
if fingerprints_clean['ridge_density'].max() > 1.0:
    fingerprints_clean['ridge_density'] = (fingerprints_clean['ridge_density'] - fingerprints_clean['ridge_density'].min()) / (fingerprints_clean['ridge_density'].max() - fingerprints_clean['ridge_density'].min())

print(f"Ridge density range after normalization: {fingerprints_clean['ridge_density'].min():.3f} - {fingerprints_clean['ridge_density'].max():.3f}")

# Define age groups and their characteristics
def categorize_age_by_fingerprint_features(row):
    """
    Categorize age based on fingerprint characteristics:
    - Children (0-12 years): smaller ridge density, smaller overall size
    - Young adults (13-40 years): medium ridge density, medium size
    - Middle-aged (41-65 years): higher ridge density, larger size
    - Elderly (65+ years): lower ridge clarity, but approximate adult size
    """
    ridge_density = row['ridge_density']
    minutiae_count = row['minutiae_count']
    
    # Use both ridge density and minutiae count for better categorization
    # Normalize minutiae count to 0-1 scale
    max_minutiae = fingerprints_clean['minutiae_count'].max()
    normalized_minutiae = minutiae_count / max_minutiae if max_minutiae > 0 else 0
    
    # Combined score: 70% ridge density + 30% minutiae count
    combined_score = (ridge_density * 0.7) + (normalized_minutiae * 0.3)
    
    if combined_score < 0.3:  # Low score - likely children
        return 'child'
    elif combined_score < 0.6:  # Medium score - likely young adults
        return 'young_adult'
    elif combined_score < 0.8:  # High score - likely middle-aged
        return 'middle_aged'
    else:  # Very high score - likely elderly
        return 'elderly'

# Apply age categorization
fingerprints_clean['age_category'] = fingerprints_clean.apply(categorize_age_by_fingerprint_features, axis=1)

# Define sex assignment based on fingerprint characteristics
def assign_sex_by_fingerprint_features(row):
    """
    Assign sex based on fingerprint characteristics:
    - Males: typically larger ridge density, wider patterns
    - Females: typically intermediate ridge density, slimmer patterns
    """
    ridge_density = row['ridge_density']
    minutiae_count = row['minutiae_count']
    
    # Calculate median values for comparison
    median_ridge = fingerprints_clean['ridge_density'].median()
    median_minutiae = fingerprints_clean['minutiae_count'].median()
    
    # Use ridge density and minutiae count to determine sex
    # Higher values typically indicate male characteristics
    if ridge_density > median_ridge and minutiae_count > median_minutiae:
        return 1  # Male
    else:
        return 2  # Female

# Apply sex assignment
fingerprints_clean['Sex'] = fingerprints_clean.apply(assign_sex_by_fingerprint_features, axis=1)

# Step 2: Map age categories to realistic age ranges and sample from appropriate NHANES subgroups
def assign_realistic_age(row):
    """
    Assign realistic age based on age category and sex,
    sampling from appropriate NHANES subgroups
    """
    age_category = row['age_category']
    sex = row['Sex']
    
    # Filter NHANES data by sex
    sex_nhanes = nhanes[nhanes['RIAGENDR'] == sex]
    
    if age_category == 'child':
        # Children: 0-12 years
        age_subgroup = sex_nhanes[sex_nhanes['RIDAGEYR'] <= 12]
        if len(age_subgroup) > 0:
            return np.random.choice(age_subgroup['RIDAGEYR'])
        else:
            return np.random.randint(0, 13)
    
    elif age_category == 'young_adult':
        # Young adults: 13-40 years
        age_subgroup = sex_nhanes[(sex_nhanes['RIDAGEYR'] >= 13) & (sex_nhanes['RIDAGEYR'] <= 40)]
        if len(age_subgroup) > 0:
            return np.random.choice(age_subgroup['RIDAGEYR'])
        else:
            return np.random.randint(13, 41)
    
    elif age_category == 'middle_aged':
        # Middle-aged: 41-65 years
        age_subgroup = sex_nhanes[(sex_nhanes['RIDAGEYR'] >= 41) & (sex_nhanes['RIDAGEYR'] <= 65)]
        if len(age_subgroup) > 0:
            return np.random.choice(age_subgroup['RIDAGEYR'])
        else:
            return np.random.randint(41, 66)
    
    else:  # elderly
        # Elderly: 65+ years
        age_subgroup = sex_nhanes[sex_nhanes['RIDAGEYR'] >= 65]
        if len(age_subgroup) > 0:
            return np.random.choice(age_subgroup['RIDAGEYR'])
        else:
            return np.random.randint(65, 85)

# Apply realistic age assignment
fingerprints_clean['RIDAGEYR'] = fingerprints_clean.apply(assign_realistic_age, axis=1)

# Step 3: Convert age to months safely
fingerprints_clean['AgeMonths'] = (fingerprints_clean['RIDAGEYR'] * 12).round().astype('Int64')

# Step 4: Merge NHANES nutritional outcomes using the realistic age/sex assignments
print("Merging with NHANES nutritional data...")
combined_df = pd.merge(
    fingerprints_clean,
    nhanes,
    left_on=['Sex', 'RIDAGEYR'],
    right_on=['RIAGENDR', 'RIDAGEYR'],
    how='left'
)

# Step 5: Merge WHO child Z-scores (only for children <=60 months)
print("Merging with WHO child growth standards...")
who_children = who[who['AgeRange'] == '0-5y'].copy()

# Ensure proper data types for merging
who_children['AgeMonths'] = who_children['AgeMonths'].round().astype('Int64')
who_children['Sex'] = who_children['Sex'].astype(str)  # Convert to string for consistent merging

# Convert our Sex column to string for consistent merging
combined_df['Sex'] = combined_df['Sex'].astype(str)

# Only merge WHO data for children (<=60 months)
children_mask = combined_df['AgeMonths'] <= 60
children_df = combined_df[children_mask].copy()

if len(children_df) > 0:
    # Merge WHO data for children
    children_with_who = pd.merge_asof(
        children_df.sort_values('AgeMonths'),
    who_children.sort_values('AgeMonths'),
    on='AgeMonths',
    by='Sex',
    direction='nearest'
)

    # Update the combined_df with WHO data for children
    combined_df.loc[children_mask] = children_with_who
else:
    print("No children <=60 months found in the dataset")

# Convert Sex back to numeric for consistency
combined_df['Sex'] = pd.to_numeric(combined_df['Sex'], errors='coerce')

# Step 6: Add summary statistics for verification
print("\n=== Assignment Summary ===")
print(f"Total records: {len(combined_df)}")
print("\nAge Category Distribution:")
print(combined_df['age_category'].value_counts())
print("\nSex Distribution:")
print(combined_df['Sex'].value_counts())
print("\nAge Range by Category:")
for category in combined_df['age_category'].unique():
    ages = combined_df[combined_df['age_category'] == category]['RIDAGEYR']
    print(f"{category}: {ages.min()}-{ages.max()} years (mean: {ages.mean():.1f})")

# Step 7: Save the combined dataset
combined_df.to_csv(output_file, index=False)
print(f"\n[OK] Combined dataset saved: {output_file}")
print(f"Final dataset: {len(combined_df)} rows, {combined_df.shape[1]} columns")

# Step 8: Verify data quality
print("\n=== Data Quality Check ===")
print(f"Missing values in key columns:")
print(combined_df[['Subject_ID', 'Sex', 'RIDAGEYR', 'AgeMonths', 'ridge_density', 'minutiae_count']].isnull().sum())

print("\n[SUCCESS] Fingerprint-based nutrition dataset created with realistic age/sex assignments!")

