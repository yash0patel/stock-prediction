import sys
import os
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Parameters
TEST_SIZE = 0.2
N_SPLITS = 5
RANDOM_STATE = 42
MIN_ROWS = 60  # Minimum rows required to train

def select_features(df):
    """Select all numeric feature columns except identifiers and targets."""
    exclude = {'Date', 'Movement', 'NextClose', 'ReturnNext'}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Movement'])
    if len(df) < MIN_ROWS:
        logger.warning(f"{os.path.basename(filepath)} skipped: only {len(df)} rows")
        return None, None, None
    features = select_features(df)
    return df[features], df['Movement'], features

def prepare(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_enc[:split], y_enc[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, le, scaler

def build_models():
    models = []
    models.append(('RandomForest', RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)))
    if XGBOOST_AVAILABLE:
        models.append(('XGBoost', xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE)))
    models.append(('LogisticRegression', LogisticRegression(
        max_iter=2000, class_weight='balanced',
        solver='liblinear', random_state=RANDOM_STATE)))
    return models

def evaluate(name, model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"{name} accuracy: {acc:.4f}")
    logger.info("Confusion matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    logger.info("Classification report:")
    logger.info(classification_report(y_test, y_pred, target_names=le.classes_))
    return acc

def log_feature_importance(name, model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-5:][::-1]
        logger.info(f"Top features for {name}:")
        for i in idx:
            logger.info(f"  {feature_names[i]}: {importances[i]:.4f}")

def train_one(filepath):
    ticker = os.path.basename(filepath).replace('_features.csv', '')
    logger.info(f"Training for {ticker}")
    X, y, features = load_data(filepath)
    if X is None:
        return
    X_train, X_test, y_train, y_test, le, scaler = prepare(X, y)
    best_acc, best_name, best_model = 0, None, None

    # Train base models
    for name, model in build_models():
        try:
            logger.info(f"Fitting {name}")
            if name == 'XGBoost':
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, shuffle=False)
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=10,
                          verbose=False)
            else:
                model.fit(X_train, y_train)
            acc = evaluate(name, model, X_test, y_test, le)
            log_feature_importance(name, model, features)
            if acc > best_acc:
                best_acc, best_name, best_model = acc, name, model
        except Exception as e:
            logger.error(f"Error training {name}: {e}")

    # Voting ensemble
    try:
        ensemble = VotingClassifier(
            estimators=build_models(), voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        acc = evaluate('Ensemble', ensemble, X_test, y_test, le)
        if acc > best_acc:
            best_acc, best_name, best_model = acc, 'Ensemble', ensemble
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")

    # Save best model
    model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    pd.to_pickle({'model': best_model, 'scaler': scaler, 'le': le}, model_path)
    logger.info(f"Saved best model {best_name} ({best_acc:.4f}) to {model_path}")

def main():
    warnings.filterwarnings('ignore')
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('_features.csv'):
            train_one(os.path.join(DATA_DIR, fname))
    logger.info("Training complete for all stocks")

if __name__ == '__main__':
    main()
