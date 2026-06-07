import os
import numpy as np
import pandas as pd
import cv2
from skimage.filters import gabor
from skimage.util import img_as_float
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pyreadstat

# =============================
# STEP 0: Import pygrowup2
# =============================
from pygrowup2 import GrowUp
who = GrowUp()

# =============================
# STEP 1: Load Fingerprint Dataset (SOCOFing)
# =============================
fingerprint_dir = "SOCOFing/Real"  # Ensure SOCOFing dataset is here
fingerprint_features = []

for subj_id, file in enumerate(os.listdir(fingerprint_dir)[:200]):  # limit for demo
    img_path = os.path.join(fingerprint_dir, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (256, 256))

    # Ridge density
    edges = cv2.Canny(img, 100, 200)
    ridge_density = np.sum(edges) / img.size

    # Minutiae count
    corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
    minutiae_count = len(corners) if corners is not None else 0

    # Gabor texture
    filt_real, _ = gabor(img_as_float(img), frequency=0.1)
    gabor_mean = filt_real.mean()

    fingerprint_features.append({
        "finger_id": file,
        "subj_id": subj_id,
        "ridge_density": ridge_density,
        "minutiae_count": minutiae_count,
        "gabor_mean": gabor_mean
    })

fingerprint_df = pd.DataFrame(fingerprint_features)
print("Fingerprint features sample:")
print(fingerprint_df.head())

# =============================
# STEP 2: Load NHANES DEMO + BMX
# =============================
nhanes_demo, meta_demo = pyreadstat.read_xport("DEMO_J.XPT")
nhanes_bmx, meta_bmx = pyreadstat.read_xport("BMX_J.XPT")

# Merge on SEQN
nhanes = pd.merge(nhanes_demo, nhanes_bmx, on="SEQN", how="inner")

# Keep relevant columns and rename
nhanes = nhanes[["SEQN", "RIDAGEYR", "RIAGENDR", "BMXWT", "BMXHT", "BMXBMI", "BMXARMC", "RIDEXPRG"]]
nhanes = nhanes.rename(columns={
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
    "BMXBMI": "BMI",
    "BMXARMC": "muac_cm",
    "RIDEXPRG": "pregnant_flag"
})

# Encode sex
nhanes['sex'] = nhanes['sex'].map({1: 'M', 2: 'F'})
# Encode pregnancy
nhanes['pregnant'] = nhanes['pregnant_flag'].map({1: True, 2: False})

# =============================
# STEP 3: Compute malnutrition labels for all groups using pygrowup2
# =============================
# Children <19 → WHO z-scores (WAZ, HAZ, WHZ)
children = nhanes[nhanes['age'] < 19].copy()
children['nutrition_outcome'] = 'normal'

# Compute WAZ, HAZ, WHZ using pygrowup2
children['WAZ'] = children.apply(lambda row: who.zscore(value=row['weight_kg'], age=row['age'], sex=row['sex'], measure='wfa'), axis=1)
children['HAZ'] = children.apply(lambda row: who.zscore(value=row['height_cm'], age=row['age'], sex=row['sex'], measure='hfa'), axis=1)
children['WHZ'] = children.apply(lambda row: who.zscore(value=row['weight_kg']/row['height_cm'], age=row['age'], sex=row['sex'], measure='wfh'), axis=1)

children.loc[children['WAZ'] < -2, 'nutrition_outcome'] = 'underweight'
children.loc[children['HAZ'] < -2, 'nutrition_outcome'] = 'stunted'
children.loc[children['WHZ'] < -2, 'nutrition_outcome'] = 'wasted'

# Adults >=19 → BMI categories
adults = nhanes[nhanes['age'] >= 19].copy()
adults['nutrition_outcome'] = pd.cut(adults['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['underweight','normal','overweight','obese'])

# Pregnant women → flag at-risk if underweight by BMI or MUAC
adults.loc[adults['pregnant'] == True, 'nutrition_outcome'] = adults.apply(lambda row: 'at_risk' if row['BMI'] < 18.5 or row['muac_cm'] < 23 else row['nutrition_outcome'], axis=1)

# Combine children + adults (with pregnant adjustments)
hanes_final = pd.concat([children, adults], ignore_index=True)

# =============================
# STEP 4: Stratified Synthetic Pairing with fingerprints
# =============================
hanes_final['age_band'] = pd.cut(nhanes_final['age'], bins=[0,18,40,60,100], labels=['child','young_adult','adult','senior'])
min_len = min(len(fingerprint_df), len(nhanes_final))
paired_df = fingerprint_df.iloc[:min_len].reset_index(drop=True)
hanes_final = nhanes_final.sample(min_len).reset_index(drop=True)
combined_df = pd.concat([paired_df, nhanes_final], axis=1)

# =============================
# STEP 5: Save combined dataset
# =============================
combined_df.to_csv('combined_fingerprint_nutrition.csv', index=False)
print("Combined dataset saved as 'combined_fingerprint_nutrition.csv'")

# =============================
# STEP 6: Train Baseline Classifier
# =============================
X = combined_df[['ridge_density','minutiae_count','gabor_mean']]
y = combined_df['nutrition_outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))

# =============================
# STEP 7: Save predictions
# =============================
test_results = X_test.copy()
test_results['true_outcome'] = y_test.values
test_results['predicted_outcome'] = y_pred
test_results.to_csv('fingerprint_nutrition_predictions.csv', index=False)
print("Predictions saved as 'fingerprint_nutrition_predictions.csv'")