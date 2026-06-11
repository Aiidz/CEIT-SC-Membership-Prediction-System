import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw CEIT-SC enrollment and collection data."""
    df = df.copy()
    
    df = df.dropna(subset=['population', 'paid_memberships'])
    
    fill_zero_cols = ['online_payments', 'facetf_payments', 'benefits_claimed', 'officer_count', 'events_held']
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    if 'online_payments' in df.columns and 'facetf_payments' in df.columns:
        total_payments = df['online_payments'] + df['facetf_payments']
        df['payment_ratio'] = np.where(total_payments > 0, df['online_payments'] / total_payments, 0.0)
    else:
        df['payment_ratio'] = 0.0
    
    if 'semester' in df.columns:
        df['semester_indicator'] = df['semester'].astype(str).str.lower().str.strip().map({
            '1st': 1, '1': 1, 'first': 1, '1st sem': 1,
            '2nd': 0, '0': 0, 'second': 0, '2nd sem': 0
        }).fillna(0).astype(int)
    else:
        df['semester_indicator'] = 0
        
    return df

if __name__ == "__main__":
    import os
    raw_path = "data/ceitsc_raw.csv"
    cleaned_path = "data/ceitsc_cleaned.csv"
    
    if os.path.exists(raw_path):
        print(f"Loading raw data from {raw_path}...")
        raw_df = pd.read_csv(raw_path)
        clean_df = preprocess_data(raw_df)
        os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
        clean_df.to_csv(cleaned_path, index=False)
        print(f"Successfully preprocessed and saved cleaned data to {cleaned_path} ({len(clean_df)} rows).")
    else:
        print(f"Raw data file not found at {raw_path}. Run generating scripts or upload data.")
