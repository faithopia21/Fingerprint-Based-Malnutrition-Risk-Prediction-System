#!/usr/bin/env python3
"""
Baseline Malnutrition Risk Prediction Classifier - VS Code Fixed Version
=======================================================================

This script implements a baseline logistic regression classifier for predicting
malnutrition risk using fingerprint features and nutritional outcomes.

Features:
- Reads fingerprint_nutrition.csv dataset
- Creates malnutrition target variable from BMI-for-age and other indicators
- Trains logistic regression on fingerprint features
- Evaluates model performance with multiple metrics
- Generates comprehensive predictions output

Output:
- fingerprint_nutrition_predictions.csv with all required columns
- Model performance metrics and evaluation
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to Python path for VS Code compatibility
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    from sklearn.impute import SimpleImputer
    print("✅ All required packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore')

def check_working_directory():
    """Check and display current working directory information"""
    print("=== Working Directory Check ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent.absolute()}")
    print(f"Files in current directory:")
    
    try:
        files = os.listdir('.')
        for file in files:
            if file.endswith('.py') or file.endswith('.csv'):
                print(f"  📄 {file}")
    except Exception as e:
        print(f"  ❌ Error listing files: {e}")
    
    # Check if required files exist
    required_files = ['fingerprint_nutrition.csv']
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file} - Found")
        else:
            print(f"  ❌ {file} - Missing")
    
    print()

def create_malnutrition_target(df):
    """
    Create malnutrition target variable based on age-appropriate thresholds:
    - Children (≤18 years): BMI-for-age Z-scores, WHZ, WAZ
    - Adults (>18 years): BMI thresholds, MUAC
    """
    print("=== Creating Malnutrition Target Variable ===")
    
    # Initialize malnutrition column
    df['Malnutrition'] = 0
    
    # Separate children and adults
    children_mask = df['Assigned_Age'] <= 18
    adults_mask = df['Assigned_Age'] > 18
    
    print(f"Children (≤18 years): {children_mask.sum()} participants")
    print(f"Adults (>18 years): {adults_mask.sum()} participants")
    
    # For children: Use BMI-for-age and other anthropometric measures
    if children_mask.sum() > 0:
        children_df = df[children_mask].copy()
        
        # Create malnutrition indicators for children
        # Underweight: BMI < 18.5 or very low for age
        underweight_mask = (
            (children_df['BMXBMI'] < 18.5) |
            (children_df['BMXBMI'] < 16.0) |  # Severe underweight
            (children_df['BMXWT'].notna() & children_df['BMXHT'].notna() & 
             (children_df['BMXWT'] / (children_df['BMXHT']/100)**2 < 16.0))
        )
        
        # Stunting: Height-for-age issues (very short for age)
        stunting_mask = (
            (children_df['BMXHT'].notna()) &
            (children_df['Assigned_Age'] <= 5) &
            (children_df['BMXHT'] < 100)  # Very short for young children
        )
        
        # Wasting: Weight-for-height issues
        wasting_mask = (
            (children_df['BMXWT'].notna() & children_df['BMXHT'].notna()) &
            (children_df['BMXWT'] / (children_df['BMXHT']/100)**2 < 16.0)
        )
        
        # Mark children as malnourished if any condition is met
        malnutrition_children = underweight_mask | stunting_mask | wasting_mask
        df.loc[children_mask, 'Malnutrition'] = malnutrition_children.astype(int)
        
        print(f"Children malnutrition cases: {malnutrition_children.sum()}")
    
    # For adults: Use BMI thresholds and other measures
    if adults_mask.sum() > 0:
        adults_df = df[adults_mask].copy()
        
        # Underweight: BMI < 18.5
        underweight_adults = adults_df['BMXBMI'] < 18.5
        
        # Overweight/Obese: BMI > 25 (not malnutrition but health risk)
        overweight_adults = adults_df['BMXBMI'] > 25
        
        # Mark adults as malnourished if underweight
        df.loc[adults_mask, 'Malnutrition'] = underweight_adults.astype(int)
        
        print(f"Adult malnutrition cases: {underweight_adults.sum()}")
        print(f"Adult overweight cases: {overweight_adults.sum()}")
    
    # Overall malnutrition statistics
    total_malnutrition = df['Malnutrition'].sum()
    print(f"\nTotal malnutrition cases: {total_malnutrition}")
    print(f"Malnutrition prevalence: {total_malnutrition/len(df)*100:.1f}%")
    
    return df

def prepare_features(df):
    """
    Prepare fingerprint features for model training
    """
    print("\n=== Preparing Features ===")
    
    # Select fingerprint features
    fingerprint_features = [
        'ridge_density', 'minutiae_count', 'size_score'
    ]
    
    # Create additional engineered features
    df['ridge_minutiae_ratio'] = df['ridge_density'] / (df['minutiae_count'] + 1)
    df['fingerprint_complexity'] = df['ridge_density'] * df['minutiae_count']
    
    # Add size category as encoded feature
    size_category_map = {'small': 0, 'medium': 1, 'large': 2}
    df['size_category_encoded'] = df['size_category'].map(size_category_map)
    
    # Add age-related features
    df['age_group'] = pd.cut(df['Assigned_Age'], 
                            bins=[0, 5, 12, 18, 30, 50, 100], 
                            labels=['infant', 'child', 'teen', 'young_adult', 'adult', 'elderly'])
    df['age_group_encoded'] = df['age_group'].cat.codes
    
    # Add sex encoding
    df['sex_encoded'] = (df['Sex'] == 'M').astype(int)
    
    # Final feature list
    feature_columns = fingerprint_features + [
        'ridge_minutiae_ratio', 'fingerprint_complexity',
        'size_category_encoded', 'age_group_encoded', 'sex_encoded'
    ]
    
    print(f"Feature columns: {feature_columns}")
    print(f"Total features: {len(feature_columns)}")
    
    return df, feature_columns

def train_baseline_model(df, feature_columns, target_column='Malnutrition'):
    """
    Train baseline logistic regression model
    """
    print("\n=== Training Baseline Model ===")
    
    # Prepare features and target
    X = df[feature_columns].copy()
    y = df[target_column]
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    print("\n=== Model Performance ===")
    
    # Training performance
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Test performance
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_prob)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    return model, scaler, imputer, feature_columns, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc
    }

def generate_predictions(df, model, scaler, imputer, feature_columns):
    """
    Generate predictions for all participants
    """
    print("\n=== Generating Predictions ===")
    
    # Prepare features for prediction
    X_pred = df[feature_columns].copy()
    
    # Handle missing values
    X_pred_imputed = pd.DataFrame(imputer.transform(X_pred), columns=X_pred.columns)
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred_imputed)
    
    # Generate predictions
    df['Malnutrition_Prob'] = model.predict_proba(X_pred_scaled)[:, 1]
    df['Predicted_Malnutrition'] = model.predict(X_pred_scaled)
    
    # Create malnutrition status labels
    df['Malnutrition_Status'] = df['Predicted_Malnutrition'].map({
        0: 'Well-nourished',
        1: 'Malnourished'
    })
    
    print(f"Predictions generated for {len(df)} participants")
    print(f"Predicted malnutrition cases: {df['Predicted_Malnutrition'].sum()}")
    print(f"Average malnutrition probability: {df['Malnutrition_Prob'].mean():.4f}")
    
    return df

def create_final_output(df):
    """
    Create final output dataset with all required columns
    """
    print("\n=== Creating Final Output Dataset ===")
    
    # Define required columns
    output_columns = [
        'Subject_ID',                    # Participant identifier
        'Participant_Number',            # Participant number
        'Sex',                           # Demographics: Sex
        'Assigned_Age',                  # Demographics: Age
        'AgeMonths',                     # Demographics: Age in months
        
        # Fingerprint features
        'ridge_density',                 # Fingerprint features
        'minutiae_count',                # Fingerprint features
        'size_score',                    # Fingerprint features
        'size_category',                 # Fingerprint features
        
        # Nutritional outcomes
        'BMXWT',                         # Weight (kg)
        'BMXHT',                         # Height (cm)
        'BMXBMI',                        # BMI
        'BMXWAIST',                      # Waist circumference
        'BMXHIP',                        # Hip circumference
        
        # Model predictions
        'Malnutrition_Prob',             # Predicted probability of malnutrition
        'Predicted_Malnutrition',        # Predicted class (0 or 1)
        'Malnutrition_Status',           # Malnutrition status (Well-nourished/Malnourished)
        
        # Target variable
        'Malnutrition'                   # Actual malnutrition status
    ]
    
    # Filter to required columns (only keep those that exist)
    existing_columns = [col for col in output_columns if col in df.columns]
    final_df = df[existing_columns].copy()
    
    print(f"Final output columns: {len(final_df.columns)}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Final dataset shape: {final_df.shape}")
    
    return final_df

def main():
    """
    Main execution function
    """
    print("Baseline Malnutrition Risk Prediction Classifier - VS Code Fixed Version")
    print("=" * 70)
    
    # Check working directory and files
    check_working_directory()
    
    # Load dataset
    print("Loading dataset...")
    try:
        # Try multiple possible paths
        possible_paths = [
            'fingerprint_nutrition.csv',
            './fingerprint_nutrition.csv',
            str(Path(__file__).parent / 'fingerprint_nutrition.csv')
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                print(f"✅ Dataset loaded from: {path}")
                break
            except FileNotFoundError:
                print(f"❌ Not found: {path}")
                continue
        
        if df is None:
            print("❌ Error: Could not find fingerprint_nutrition.csv in any location!")
            print("Please ensure the file exists in the same directory as this script.")
            return
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Data overview
    print(f"\nDataset columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Create malnutrition target variable
    df = create_malnutrition_target(df)
    
    # Prepare features
    df, feature_columns = prepare_features(df)
    
    # Train baseline model
    model, scaler, imputer, feature_columns, metrics = train_baseline_model(df, feature_columns)
    
    # Generate predictions
    df = generate_predictions(df, model, scaler, imputer, feature_columns)
    
    # Create final output
    final_output = create_final_output(df)
    
    # Save predictions
    output_file = 'fingerprint_nutrition_predictions.csv'
    try:
        final_output.to_csv(output_file, index=False)
        print(f"\n✅ [SUCCESS] Predictions saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        return
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total participants: {len(final_output)}")
    print(f"Actual malnutrition cases: {final_output['Malnutrition'].sum()}")
    print(f"Predicted malnutrition cases: {final_output['Predicted_Malnutrition'].sum()}")
    print(f"Model accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Model ROC-AUC: {metrics['test_roc_auc']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    print(f"\n🎉 [COMPLETE] Baseline classifier pipeline executed successfully!")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()





