import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import the feature engineering pipeline from Phase 2
from phase2_feature_engineering import engineer_features

def train_match_engine():
    """
    Executes Phase 3 of the ML Architecture:
    Trains a predictive model using chronological Time-Series splitting.
    """
    print("--- Starting Phase 3: The Match Engine ---")
    
    # 1. Fetch the engineered dataset from Phase 2
    df = engineer_features()
    
    # 2. Chronological Ordering (CRITICAL for sports ML)
    # We must sort by date so we train on the past and test on the future.
    df = df.sort_values('date').reset_index(drop=True)
    
    # 3. Define Features (X) and Target (y)
    features = ['elo_diff', 'form_diff', 'attack_diff', 'is_home_advantage']
    X = df[features]
    y = df['target']
    
    # 4. Time-Series Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    train_start, train_end = df['date'].iloc[0].date(), df['date'].iloc[split_idx-1].date()
    test_start, test_end = df['date'].iloc[split_idx].date(), df['date'].iloc[-1].date()
    
    print(f"\nTime-Series Split Successful:")
    print(f"TRAINING on {len(X_train)} matches (from {train_start} to {train_end})")
    print(f"TESTING  on {len(X_test)} matches (from {test_start} to {test_end})")
    
    # 5. Initialize the Model
    # We use 'balanced' class weights because 'Draws' (Class 1) happen less often 
    # than Wins/Losses, and we want the model to actively look for Draw conditions.
    print("\nTraining the Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,          # Number of trees
        max_depth=6,               # Keep it shallow to prevent overfitting noise
        min_samples_split=15,
        class_weight='balanced',   # Penalize missing 'Draws'
        random_state=42
    )
    
    # 6. Train the Model
    model.fit(X_train, y_train)
    
    # 7. Evaluate Performance
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification Report (0 = Away Win, 1 = Draw, 2 = Home Win):")
    print(classification_report(y_test, y_pred))
    
    # 8. Feature Importance (What drives the predictions?)
    print("\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feature, importance in importances.items():
        print(f"{feature:>18}: {importance:.1%}")
        
    print("\n--- Phase 3 Complete ---")
    
    return model, df

if __name__ == "__main__":
    trained_model, final_df = train_match_engine()