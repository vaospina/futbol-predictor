"""
Modelo 1X2 - Prediccion de resultado del partido.
XGBoost Classifier multiclase (Home/Draw/Away).
"""
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from models.feature_engineering import MATCH_FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "match_predictor_1x2.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_1x2.joblib")

LABELS = ["home_win", "draw", "away_win"]


class MatchPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABELS)
        self.feature_names = MATCH_FEATURE_NAMES

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Entrena el modelo 1X2.
        X: matriz de features ordenada cronológicamente (asc).
        y: array de labels ('home_win', 'draw', 'away_win')
        Retorna metricas de cross-validation.
        """
        y_encoded = self.label_encoder.transform(y)

        base_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,          # reducido de 6 → menos overfit
            learning_rate=0.05,   # reducido → mejor generalización
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,       # L2 regularización
            reg_alpha=0.5,        # L1 regularización
            min_child_weight=5,   # evita splits sobre muy pocos samples
            gamma=0.1,            # mínima reducción de loss para un split
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
        )

        # Métricas con TimeSeriesSplit (sin calibración para CV puro)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_accuracy = cross_val_score(base_model, X, y_encoded, cv=tscv, scoring="accuracy")
        cv_f1 = cross_val_score(base_model, X, y_encoded, cv=tscv, scoring="f1_macro")
        cv_logloss = cross_val_score(base_model, X, y_encoded, cv=tscv, scoring="neg_log_loss")

        # Entrenar modelo base en 85% → calibrar Platt scaling en el 15% más reciente
        # Esto corrige la sobreconfianza de XGBoost en probabilidades multiclase
        cal_split = max(int(len(X) * 0.85), len(X) - 200)
        self._base_model = base_model
        self._base_model.fit(X[:cal_split], y_encoded[:cal_split])

        self.model = CalibratedClassifierCV(self._base_model, cv="prefit", method="sigmoid")
        self.model.fit(X[cal_split:], y_encoded[cal_split:])

        metrics = {
            "accuracy_cv": float(np.mean(cv_accuracy)),
            "f1_score": float(np.mean(cv_f1)),
            "log_loss": float(-np.mean(cv_logloss)),
            "training_samples": len(X),
        }

        logger.info(
            f"Modelo 1X2 entrenado (calibrado): accuracy={metrics['accuracy_cv']:.3f}, "
            f"f1={metrics['f1_score']:.3f}, logloss={metrics['log_loss']:.3f}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> list:
        """
        Predice resultado para cada partido.
        Retorna lista de dicts con label y probabilidades.
        """
        if self.model is None:
            logger.error("Modelo 1X2 no entrenado/cargado")
            return []

        probas = self.model.predict_proba(X)
        results = []
        for proba in probas:
            idx = np.argmax(proba)
            label = self.label_encoder.inverse_transform([idx])[0]
            results.append({
                "prediction": label,
                "probability": float(proba[idx]),
                "probabilities": {
                    "home_win": float(proba[0]),
                    "draw": float(proba[1]),
                    "away_win": float(proba[2]),
                },
            })
        return results

    def save(self, path: str = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or MODEL_PATH
        joblib.dump(self.model, path)
        joblib.dump(self.label_encoder, ENCODER_PATH)
        base_path = path.replace(".joblib", "_base.joblib")
        if hasattr(self, "_base_model") and self._base_model is not None:
            joblib.dump(self._base_model, base_path)
        logger.info(f"Modelo 1X2 guardado en {path}")

    def load(self, path: str = None):
        path = path or MODEL_PATH
        if os.path.exists(path):
            self.model = joblib.load(path)
            if os.path.exists(ENCODER_PATH):
                self.label_encoder = joblib.load(ENCODER_PATH)
            base_path = path.replace(".joblib", "_base.joblib")
            if os.path.exists(base_path):
                self._base_model = joblib.load(base_path)
            logger.info("Modelo 1X2 cargado")
            return True
        logger.warning(f"No se encontro modelo 1X2 en {path}")
        return False

    def get_feature_importance(self) -> dict:
        # CalibratedClassifierCV no expone feature_importances_ directamente;
        # usar el base_model entrenado antes de calibrar.
        source = getattr(self, "_base_model", None) or self.model
        if source is None:
            return {}
        importance = getattr(source, "feature_importances_", None)
        if importance is None:
            return {}
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
