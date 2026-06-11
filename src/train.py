import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import pickle
import os

def train_model(df: pd.DataFrame, model_dir: str = "models") -> dict:
    """Train MLR model, save it, and return metrics + OLS summary."""
    features = [
        'population', 
        'payment_ratio', 
        'semester_indicator', 
        'benefits_claimed', 
        'officer_count', 
        'events_held'
    ]
    
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Required feature column '{f}' is missing from the dataset.")
            
    if 'paid_memberships' not in df.columns:
        raise ValueError("Target column 'paid_memberships' is missing from the dataset.")
        
    X = df[features]
    y = df['paid_memberships']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ceitsc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    
    return {
        "model_path": model_path,
        "scikit_model": model,
        "ols_model": ols_model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "features": features
    }

if __name__ == "__main__":
    cleaned_path = "data/ceitsc_cleaned.csv"
    if os.path.exists(cleaned_path):
        print(f"Loading cleaned data from {cleaned_path}...")
        df = pd.read_csv(cleaned_path)
        results = train_model(df)
        
        print("\n=== scikit-learn Model Evaluation ===")
        print(f"Model saved to: {results['model_path']}")
        print(f"R² (Coefficient of Determination): {results['r2']:.4f}")
        print(f"Mean Absolute Error (MAE):         {results['mae']:.2f}")
        print(f"Root Mean Squared Error (RMSE):    {results['rmse']:.2f}")
        print(f"Intercept:                         {results['intercept']:.4f}")
        print("Coefficients:")
        for feat, coef in zip(results['features'], results['coefficients']):
            print(f"  {feat}: {coef:.4f}")
            
        print("\n=== statsmodels OLS Regression Results ===")
        print(results['ols_model'].summary())
    else:
        print(f"Cleaned dataset not found at {cleaned_path}. Run preprocess script first.")
