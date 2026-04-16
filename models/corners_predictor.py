"""
Modelo Corners Over/Under.
XGBoost Regressor que predice el total de corners de un partido.
Luego se compara con la linea de la casa de apuestas.
"""
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import norm
from models.feature_engineering import CORNERS_FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "corners_predictor.joblib")


class CornersPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = CORNERS_FEATURE_NAMES
        self.residual_std = 3.0  # se ajusta despues del entrenamiento

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Entrena el modelo de corners.
        X: features, y: total corners reales del partido
        """
        self.model = XGBRegressor(
            n_estimators=150,
            max_depth=5,
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

        # Calcular residual std para probabilidades
        predictions = self.model.predict(X)
        self.residual_std = float(np.std(y - predictions))

        metrics = {
            "r2_cv": float(np.mean(cv_r2)),
            "mae_cv": float(-np.mean(cv_mae)),
            "residual_std": self.residual_std,
            "training_samples": len(X),
        }

        logger.info(
            f"Modelo Corners entrenado: R2={metrics['r2_cv']:.3f}, "
            f"MAE={metrics['mae_cv']:.2f}"
        )
        return metrics

    def predict(self, X: np.ndarray, lines: list = None) -> list:
        """
        Predice total corners y probabilidades over/under.
        lines: lista de lineas (ej: [9.5, 10.5]) para cada partido.
        """
        if self.model is None:
            logger.error("Modelo corners no entrenado/cargado")
            return []

        predicted_corners = self.model.predict(X)
        results = []

        for i, pred in enumerate(predicted_corners):
            line = lines[i] if lines and i < len(lines) else 9.5

            # Probabilidad de over usando distribucion normal
            prob_over = 1 - norm.cdf(line, loc=pred, scale=self.residual_std)
            prob_under = 1 - prob_over

            if prob_over >= prob_under:
                prediction = f"over_{line}"
                probability = prob_over
            else:
                prediction = f"under_{line}"
                probability = prob_under

            results.append({
                "predicted_corners": float(pred),
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
        joblib.dump({"model": self.model, "residual_std": self.residual_std}, path)
        logger.info(f"Modelo corners guardado en {path}")

    def load(self, path: str = None):
        path = path or MODEL_PATH
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]
            self.residual_std = data.get("residual_std", 3.0)
            logger.info("Modelo corners cargado")
            return True
        logger.warning(f"No se encontro modelo corners en {path}")
        return False

    def get_feature_importance(self) -> dict:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
