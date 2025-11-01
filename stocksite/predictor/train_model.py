import sys
import os
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import xgboost as xgb
    from xgboost.callback import EarlyStopping
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ROWS = 60  # Minimum rows required to train

def select_features(df):
    """Select numeric features excluding identifiers and targets."""
    exclude = {'Date', 'Movement', 'NextClose', 'ReturnNext'}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def load_data(filepath):
    """Load features and target from CSV, validate minimal rows."""
    if not os.path.exists(filepath):
        logger.error(f"Feature file not found: {filepath}")
        return None, None, None
    df = pd.read_csv(filepath)
    if 'Movement' not in df.columns:
        logger.error(f"'Movement' column missing in {filepath}")
        return None, None, None
    df = df.dropna(subset=['Movement'])
    if len(df) < MIN_ROWS:
        logger.warning(f"{os.path.basename(filepath)} skipped: only {len(df)} rows (minimum {MIN_ROWS})")
        return None, None, None
    features = select_features(df)
    if not features:
        logger.error(f"No valid features found in {filepath}")
        return None, None, None
    X = df[features]
    y = df['Movement']
    logger.info(f"Loaded {len(df)} rows with {len(features)} features from {os.path.basename(filepath)}")
    return X, y, features

def prepare_data(X, y):
    """Encode labels, split into train/test, and scale features."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # Time-based split
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler

def build_base_models():
    """Return a list of base models."""
    models = [
        ('RandomForest', RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1))
    ]
    if XGBOOST_AVAILABLE:
        models.append(('XGBoost', xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE)))
    models.append(('LogisticRegression', LogisticRegression(
        max_iter=1000, class_weight='balanced',
        solver='liblinear', random_state=RANDOM_STATE)))
    return models

def evaluate_model(name, model, X_test, y_test, le):
    """Evaluate and log performance metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"{name} Accuracy: {acc:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=le.classes_))
    return acc

def log_top_features(name, model, feature_names, top_n=5):
    """Log top feature importances if available."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        logger.info(f"Top {top_n} features for {name}:")
        for idx in indices:
            logger.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")

def train_one(filepath):
    """Train models for a single stock feature file and save the best."""
    ticker = os.path.basename(filepath).replace('_features.csv', '')
    logger.info(f"\n--- Training for {ticker} ---")
    X, y, features = load_data(filepath)
    if X is None:
        return

    X_train, X_test, y_train, y_test, le, scaler = prepare_data(X, y)
    best_acc, best_model, best_name = 0, None, None

    # Train base models
    for name, model in build_base_models():
        try:
            logger.info(f"Fitting {name}...")
            if name == 'XGBoost':
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, shuffle=False)
                # Try modern early stopping API, fallback to older, then to no ES
                try:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[EarlyStopping(rounds=10)],
                        verbose=False
                    )
                except TypeError:
                    try:
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=10,
                            verbose=False
                        )
                    except TypeError:
                        model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            acc = evaluate_model(name, model, X_test, y_test, le)
            log_top_features(name, model, features)
            if acc > best_acc:
                best_acc, best_model, best_name = acc, model, name
        except Exception as e:
            logger.error(f"Error training {name}: {e}")

    # Train Voting Ensemble
    try:
        estimators = build_base_models()
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        acc = evaluate_model('Ensemble', ensemble, X_test, y_test, le)
        if acc > best_acc:
            best_acc, best_model, best_name = acc, ensemble, 'Ensemble'
    except Exception as e:
        logger.error(f"Error training Ensemble: {e}")

    # Save best model using joblib.dump to ensure correct dict serialization
    if best_model is not None:
        model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        joblib.dump({'model': best_model, 'scaler': scaler, 'le': le}, model_path)
        logger.info(f"Saved best model {best_name} (Acc: {best_acc:.4f}) to {model_path}")
    else:
        logger.warning(f"No model was trained successfully for {ticker}")

def main():
    """Main entry point: train on all feature CSVs or specific symbol."""
    warnings.filterwarnings('ignore')
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        feature_file = os.path.join(DATA_DIR, f"{symbol}_features.csv")
        if os.path.exists(feature_file):
            train_one(feature_file)
        else:
            logger.error(f"Features file not found for {symbol}: {feature_file}")
    else:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')]
        if not files:
            logger.error("No feature files found. Run prepare_features.py first.")
            return
        for fname in files:
            train_one(os.path.join(DATA_DIR, fname))
        logger.info("\nTraining complete for all stocks.")

if __name__ == '__main__':
    main()
