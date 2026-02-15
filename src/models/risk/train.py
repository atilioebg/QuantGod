import argparse
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json
import logging
from pathlib import Path
from sklearn.metrics import classification_report, precision_score
from datetime import datetime
import sys
import joblib

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.config import settings

# Setup Logging
def setup_logger():
    log_dir = Path("logs/risk")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"xgb_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("RiskManager")

logger = setup_logger()

def load_and_merge_data():
    """
    Loads meta-features and targets from 'data/processed/meta/'.
    Merges on timestamp (chronological).
    """
    data_dir = settings.PROCESSED_DIR / "meta"
    
    logger.info("Scanning for data files...")
    meta_files = sorted(list(data_dir.glob("meta_*.parquet")))
    target_files = sorted(list(data_dir.glob("target_*.csv")))
    
    if not meta_files or not target_files:
        raise FileNotFoundError(f"No data found in {data_dir}. Run precompute_tensors.py first!")
    
    logger.info(f"Found {len(meta_files)} meta files and {len(target_files)} target files.")
    
    # Load all (Lazy if big, but distinct files)
    # Using pandas for ease with XGBoost/Sklearn for now, or Polars for speed then convert.
    # Meta is Parquet (fast), Target is CSV.
    
    dfs = []
    
    for mf, tf in zip(meta_files, target_files):
        # Check alignment by filename roughly? 
        # Assumes sorted(glob) aligns them if naming is consistent "meta_2023-01", "target_2023-01"
        m_month = mf.stem.replace("meta_", "")
        t_month = tf.stem.replace("target_", "")
        
        if m_month != t_month:
            logger.warning(f"Mismatch filenames: {mf.name} vs {tf.name}. Skipping pairing.")
            continue
            
        logger.info(f"Loading {m_month}...")
        
        # Load
        df_meta = pd.read_parquet(mf)
        df_target = pd.read_csv(tf)
        
        # Merge
        # Ensure join keys are strings or datetime64
        # meta parquet might have 'timestamp_str' or 'datetime'
        # target csv has 'timestamp'
        
        if "timestamp_str" in df_meta.columns:
            left_on = "timestamp_str"
        elif "datetime" in df_meta.columns:
            # Convert to match target format if needed, but string join is safest
            df_meta["timestamp_str"] = df_meta["datetime"].astype(str)
            left_on = "timestamp_str"
        else:
            raise ValueError(f"No timestamp column in {mf}")
            
        df_target["timestamp"] = df_target["timestamp"].astype(str)
        
        merged = pd.merge(df_meta, df_target, left_on=left_on, right_on="timestamp", how="inner")
        dfs.append(merged)
        
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Sort by time just in case
    if "datetime" in full_df.columns:
        full_df.sort_values("datetime", inplace=True)
    else:
        full_df.sort_values("timestamp", inplace=True)
        
    logger.info(f"Total Combined Rows: {len(full_df)}")
    return full_df

def feature_engineering_pipeline(df: pd.DataFrame):
    """
    Enhanced tabular FE for Tree Models: Lags, Deltas.
    """
    logger.info("Starting Feature Engineering...")
    
    # Numeric cols only for lags (exclude timestamps, labels)
    exclude_cols = ["timestamp", "timestamp_str", "datetime", "label", "close"] # close kept for ref, not feature?
    # Usually we use indicators as features.
    
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Replace infs/nans logic first
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Add Lags and Deltas
    # NOTE: Lags across concatenated chunks might bleed? 
    # Ideally should process lags per contiguous session or ensure sorted.
    # We sorted chronologically. Small bleed at month boundaries is minimal risk for this scale.
    
    lags = [1, 2, 3]
    
    for col in features:
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            
        # Delta (Diff)
        df[f"{col}_diff1"] = df[col].diff(1)
            
    # Dropping initial NaNs from shifting
    df.dropna(inplace=True)
    
    logger.info(f"Feature Set Size: {len(df.columns)}")
    return df

def train_risk_manager():
    # 1. Ingestion
    raw_df = load_and_merge_data()
    
    # 2. FE
    df = feature_engineering_pipeline(raw_df)
    
    # Define X and Y
    ignore_cols = ["timestamp", "timestamp_str", "datetime", "label", "close"] # close is reference
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols]
    y = df["label"]
    
    # 3. Split (80/20 Chronological)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train Size: {len(X_train)} | Val Size: {len(X_val)}")
    
    # Compute Selection Weights (Paranoid Class 1 & 3)
    # We want XGBoost to care SUPER MUCH about Class 1 (Stop) and Class 3 (Super Long)
    # Class weights: 0:1, 1:10, 2:1, 3:5 (Example)
    # Can also use sample_weight in fit()
    
    sample_weights_train = np.ones(len(y_train))
    sample_weights_train[y_train == 1] = 10.0 # STOP IS CRITICAL
    sample_weights_train[y_train == 3] = 5.0  # SUPER LONG IS MONEY
    
    # 4. Bayesian Optimization (Optuna)
    logger.info("Starting Optuna Optimization...")
    
    def objective(trial):
        param = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss', # or merror
            'booster': 'gbtree',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'verbosity': 0
        }
        
        # Train small XGB
        model = xgb.XGBClassifier(**param, n_estimators=500, early_stopping_rounds=20)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights_train,
            verbose=False
        )
        
        preds = model.predict(X_val)
        
        # Metric: Precision of Class 1 (STOP)
        # Avoid zero division
        precisions = precision_score(y_val, preds, average=None, labels=[0,1,2,3], zero_division=0)
        precision_stop = precisions[1] # Class 1
        
        return precision_stop

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30) # 30 trials
    
    best_params = study.best_params
    logger.info(f"Best Optuna Params: {best_params}")
    logger.info(f"Best Precision Class 1: {study.best_value}")
    
    # 5. Final Training
    logger.info("Training Final Model...")
    
    final_params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'tree_method': 'hist',
        'booster': 'gbtree',
        **best_params
    }
    
    model = xgb.XGBClassifier(**final_params, n_estimators=1000, early_stopping_rounds=50)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights_train,
        verbose=True
    )
    
    # 6. Audit & Report
    preds = model.predict(X_val)
    report = classification_report(y_val, preds, target_names=["Neutro", "Stop", "Long", "SuperLong"])
    
    print("\n" + "="*60)
    print("FINAL AUDIT REPORT")
    print("="*60)
    print(report)
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Paths
    artifact_dir = Path("data/artifacts/risk")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = artifact_dir / "xgb_risk_manager.json"
    fi_path = artifact_dir / "feature_importance.csv"
    report_path = Path("logs/risk/training_report.txt")
    
    # Save
    model.save_model(model_path)
    importance.to_csv(fi_path, index=False)
    
    with open(report_path, "w") as f:
        f.write("XGBoost Risk Manager Audit\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nTop 10 Features:\n")
        f.write(importance.head(10).to_string())
        
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    try:
        train_risk_manager()
    except Exception as e:
        logger.error(f"Fatal validation error: {e}")
        import traceback
        traceback.print_exc()
