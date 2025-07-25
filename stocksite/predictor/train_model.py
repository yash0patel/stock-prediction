# train_model.py - Advanced Ensemble Model Training for Indian Stock Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import logging
import time
import os
from datetime import datetime
import joblib   # <-- added for saving models

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

import matplotlib
matplotlib.use('Agg')   # add at top of train_model.py


# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import GradientBoosting
try:
    from sklearn.ensemble import GradientBoostingClassifier
    GRADIENT_BOOST_AVAILABLE = True
except ImportError:
    GRADIENT_BOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEnsembleTrainer:
    """Advanced ensemble trainer optimized for Indian stock prediction"""

    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.results = {
            'training_scores': {},
            'validation_scores': {},
            'feature_importance': {},
            'confusion_matrices': {}
        }

    def prepare_base_models(self):
        """Prepare optimized base models for ensemble"""
        base_models = []

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        base_models.append(('RandomForest', rf_model))

        # XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            base_models.append(('XGBoost', xgb_model))
        else:
            logger.warning("⚠️ XGBoost not available. Skipping XGBoost model.")

        # Gradient Boosting if available
        if GRADIENT_BOOST_AVAILABLE:
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            base_models.append(('GradientBoost', gb_model))
        else:
            logger.warning("⚠️ Gradient Boosting not available. Skipping GradientBoost model.")

        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        base_models.append(('LogisticRegression', lr_model))

        logger.info(f"✅ Prepared {len(base_models)} base models for ensemble")
        return base_models

    def create_ensemble_model(self, base_models):
        """Create stacking ensemble with meta-learner"""
        try:
            meta_learner = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
            stacking_ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )
            logger.info("✅ Created stacking ensemble model")
            return stacking_ensemble
        except Exception as e:
            logger.error(f"❌ Error creating ensemble model: {str(e)}")
            voting_ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                n_jobs=-1
            )
            logger.info("✅ Created voting ensemble as fallback")
            return voting_ensemble

    def evaluate_model(self, model, X_test, y_test, model_name):
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            self.results['validation_scores'][model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            cm = confusion_matrix(y_test, y_pred)
            self.results['confusion_matrices'][model_name] = cm
            logger.info(f"📊 {model_name} Performance: Acc={accuracy:.4f}, F1={f1:.4f}")
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
            return None

    def plot_confusion_matrix(self, cm, model_name, classes=['Up', 'Down', 'Neutral']):
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                        dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            logger.error(f"❌ Error plotting confusion matrix for {model_name}: {str(e)}")

    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                plt.figure(figsize=(10, 8))
                top_features = feature_df.head(top_n)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n} Features - {model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
                            dpi=300, bbox_inches='tight')
                plt.show()
                self.results['feature_importance'][model_name] = feature_df
            else:
                logger.info(f"ℹ️ {model_name} does not support feature importance")
        except Exception as e:
            logger.error(f"❌ Error plotting feature importance for {model_name}: {str(e)}")

    def time_series_validation(self, model, X, y, n_splits=5):
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            logger.info(f"📈 Time Series CV Scores: {scores}")
            logger.info(f"📊 Mean CV Score: {scores.mean():.4f}")
            return scores
        except Exception as e:
            logger.error(f"❌ Error in time series validation: {str(e)}")
            return np.array([])

def load_and_preprocess_data(file_path):
    try:
        logger.info(f"📂 Loading data from {file_path}")
        df = pd.read_csv(file_path)
        feature_columns = [col for col in df.columns if col not in ['Date', 'Movement', 'Future_Close', 'Future_Return']]
        X = df[feature_columns].fillna(0)
        y = df['Movement']
        mask = y.notna()
        return X[mask], y[mask], feature_columns
    except Exception as e:
        logger.error(f"❌ Error loading data from {file_path}: {str(e)}")
        return None, None, None

def train_model_for_stock(file_path):
    try:
        stock_name = os.path.basename(file_path).replace('_features.csv', '')
        logger.info(f"🎯 Training model for {stock_name}")
        X, y, feature_columns = load_and_preprocess_data(file_path)
        if X is None:
            return None

        trainer = AdvancedEnsembleTrainer()
        y_encoded = trainer.label_encoder.fit_transform(y)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        X_train_scaled = trainer.scaler.fit_transform(X_train)
        X_test_scaled = trainer.scaler.transform(X_test)

        base_models = trainer.prepare_base_models()
        trained_models = {}
        for name, model in base_models:
            logger.info(f"🏋️ Training {name}...")
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            trainer.evaluate_model(model, X_test_scaled, y_test, name)
            if hasattr(model, 'feature_importances_'):
                trainer.plot_feature_importance(model, feature_columns, name)
            trainer.time_series_validation(model, X_train_scaled, y_train)

        ensemble = None
        if len(trained_models) >= 2:
            ensemble_models = [(name, model) for name, model in trained_models.items()]
            ensemble = trainer.create_ensemble_model(ensemble_models)
            ensemble.fit(X_train_scaled, y_train)
            ensemble_results = trainer.evaluate_model(ensemble, X_test_scaled, y_test, "Ensemble")
            if ensemble_results:
                cm = confusion_matrix(y_test, ensemble_results['predictions'])
                trainer.plot_confusion_matrix(cm, "Ensemble", trainer.label_encoder.classes_)

        best_model = max(trainer.results['validation_scores'],
                         key=lambda x: trainer.results['validation_scores'][x]['accuracy'])
        best_accuracy = trainer.results['validation_scores'][best_model]['accuracy']
        logger.info(f"🏆 Best Model: {best_model} with accuracy {best_accuracy:.4f}")

        os.makedirs("models", exist_ok=True)
        best_model_instance = trained_models[best_model] if best_model != "Ensemble" else ensemble
        joblib.dump(best_model_instance, f"models/{stock_name}_model.pkl")
        logger.info(f"💾 Best model saved as models/{stock_name}_model.pkl")

        return trainer
    except Exception as e:
        logger.error(f"💥 Critical error training model for {file_path}: {str(e)}")
        return None

def main():
    start_time = time.time()
    feature_files = [f for f in os.listdir('.') if f.endswith('_features.csv')]
    if not feature_files:
        logger.error("❌ No feature files found. Run prepare_features.py first.")
        return
    logger.info(f"📁 Found {len(feature_files)} feature files")
    for file_path in feature_files:
        train_model_for_stock(file_path)
    logger.info(f"⏱️ Total execution time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()
