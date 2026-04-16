"""
Modelo Shots on Target por jugador.
XGBoost Regressor que predice shots on target de un jugador en un partido.
"""
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import poisson
from models.feature_engineering import PLAYER_FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "shots_predictor.joblib")


class ShotsPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = PLAYER_FEATURE_NAMES

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Entrena el modelo de shots on target.
        X: features del jugador + contexto partido
        y: shots on target reales
        """
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )

        tscv = TimeSeriesSplit(n_splits=5)
        cv_r2 = cross_val_score(self.model, X, y, cv=tscv, scoring="r2")
        cv_mae = cross_val_score(self.model, X, y, cv=tscv, scoring="neg_mean_absolute_error")

        self.model.fit(X, y)

        metrics = {
            "r2_cv": float(np.mean(cv_r2)),
            "mae_cv": float(-np.mean(cv_mae)),
            "training_samples": len(X),
        }

        logger.info(
            f"Modelo Shots entrenado: R2={metrics['r2_cv']:.3f}, "
            f"MAE={metrics['mae_cv']:.2f}"
        )
        return metrics

    def predict(self, X: np.ndarray, lines: list = None) -> list:
        """
        Predice shots on target y probabilidad over/under.
        Usa distribucion Poisson para calcular probabilidades.
        lines: lista de lineas (ej: [1.5, 0.5])
        """
        if self.model is None:
            logger.error("Modelo shots no entrenado/cargado")
            return []

        predicted_sot = self.model.predict(X)
        results = []

        for i, pred in enumerate(predicted_sot):
            pred = max(0, pred)
            line = lines[i] if lines and i < len(lines) else 1.5

            # Usar Poisson para probabilidades
            prob_over = 1 - poisson.cdf(int(line), mu=pred)
            prob_under = poisson.cdf(int(line), mu=pred)

            if prob_over >= prob_under:
                prediction = f"over_{line}"
                probability = prob_over
            else:
                prediction = f"under_{line}"
                probability = prob_under

            results.append({
                "predicted_sot": float(pred),
                "line": line,
                "prediction": prediction,
                "probability": float(probability),
                "prob_over": float(prob_over),
                "prob_under": float(prob_under),
            })

        return results

    def save(self, path: str = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or MODEL_PATH
        joblib.dump(self.model, path)
        logger.info(f"Modelo shots guardado en {path}")

    def load(self, path: str = None):
        path = path or MODEL_PATH
        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info("Modelo shots cargado")
            return True
        logger.warning(f"No se encontro modelo shots en {path}")
        return False

    def get_feature_importance(self) -> dict:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
