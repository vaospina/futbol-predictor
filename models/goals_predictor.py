"""
Modelo de goles - Predicciones over/under y BTTS.
XGBoost Classifier binario, una instancia por mercado.
"""
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from models.feature_engineering import MATCH_FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")


class GoalsPredictor:
    def __init__(self, market_name: str):
        self.market_name = market_name
        self.model = None
        self.feature_names = MATCH_FEATURE_NAMES
        self.model_path = os.path.join(MODEL_DIR, f"goals_{market_name}.joblib")

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )

        tscv = TimeSeriesSplit(n_splits=5)
        cv_accuracy = cross_val_score(self.model, X, y, cv=tscv, scoring="accuracy")
        cv_f1 = cross_val_score(self.model, X, y, cv=tscv, scoring="f1")
        cv_logloss = cross_val_score(self.model, X, y, cv=tscv, scoring="neg_log_loss")

        self.model.fit(X, y)

        metrics = {
            "accuracy_cv": float(np.mean(cv_accuracy)),
            "f1_score": float(np.mean(cv_f1)),
            "log_loss": float(-np.mean(cv_logloss)),
            "training_samples": len(X),
        }

        logger.info(
            f"Modelo {self.market_name} entrenado: accuracy={metrics['accuracy_cv']:.3f}, "
            f"f1={metrics['f1_score']:.3f}, logloss={metrics['log_loss']:.3f}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> list:
        if self.model is None:
            logger.error(f"Modelo {self.market_name} no entrenado/cargado")
            return []

        probas = self.model.predict_proba(X)
        results = []
        for proba in probas:
            prob_yes = float(proba[1])
            prob_no = float(proba[0])
            results.append({
                "prediction": "yes" if prob_yes >= prob_no else "no",
                "probability": max(prob_yes, prob_no),
                "prob_yes": prob_yes,
                "prob_no": prob_no,
            })
        return results

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Modelo {self.market_name} guardado en {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Modelo {self.market_name} cargado")
            return True
        logger.warning(f"No se encontro modelo {self.market_name}")
        return False

    def get_feature_importance(self) -> dict:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
