"""
Modelo Poisson para goles - predice goles esperados por equipo
y deriva probabilidades over/under y BTTS.
"""
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import poisson
from models.feature_engineering import GOALS_FEATURE_NAMES
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")


class GoalsPoissonPredictor:
    """Two XGBRegressors: one for home goals, one for away goals.
    Uses Poisson distribution to derive over/under and BTTS probabilities."""

    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.feature_names = GOALS_FEATURE_NAMES
        self.path_home = os.path.join(MODEL_DIR, "goals_poisson_home.joblib")
        self.path_away = os.path.join(MODEL_DIR, "goals_poisson_away.joblib")

    def train(self, X: np.ndarray, y_home: np.ndarray, y_away: np.ndarray) -> dict:
        params = dict(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="count:poisson",
            random_state=42,
        )

        self.model_home = XGBRegressor(**params)
        self.model_away = XGBRegressor(**params)

        tscv = TimeSeriesSplit(n_splits=5)

        cv_mae_h = cross_val_score(self.model_home, X, y_home, cv=tscv, scoring="neg_mean_absolute_error")
        cv_mae_a = cross_val_score(self.model_away, X, y_away, cv=tscv, scoring="neg_mean_absolute_error")

        self.model_home.fit(X, y_home)
        self.model_away.fit(X, y_away)

        metrics = {
            "mae_home": float(-np.mean(cv_mae_h)),
            "mae_away": float(-np.mean(cv_mae_a)),
            "training_samples": len(X),
        }
        logger.info(
            f"Poisson entrenado: MAE_home={metrics['mae_home']:.3f}, "
            f"MAE_away={metrics['mae_away']:.3f}, samples={len(X)}"
        )
        return metrics

    def predict_goals(self, X: np.ndarray) -> list:
        """Returns expected goals for each match."""
        if self.model_home is None:
            return []
        pred_h = self.model_home.predict(X)
        pred_a = self.model_away.predict(X)
        return [
            {"exp_home": max(0.01, float(h)), "exp_away": max(0.01, float(a))}
            for h, a in zip(pred_h, pred_a)
        ]

    def predict_market(self, X: np.ndarray, market: str) -> list:
        """Predict a specific market using Poisson probabilities."""
        goals = self.predict_goals(X)
        results = []
        for g in goals:
            mu_h = g["exp_home"]
            mu_a = g["exp_away"]
            mu_total = mu_h + mu_a

            if market.startswith("over_"):
                line = float(market.split("_")[1])
                prob_over = 1 - poisson.cdf(int(line), mu=mu_total)
                prob_under = 1 - prob_over
                pred = "yes" if prob_over >= 0.5 else "no"
                prob = prob_over if pred == "yes" else prob_under
                results.append({
                    "prediction": pred,
                    "probability": float(prob),
                    "prob_yes": float(prob_over),
                    "prob_no": float(prob_under),
                    "exp_total": float(mu_total),
                })
            elif market == "btts":
                # P(home>=1) * P(away>=1)
                p_home_scores = 1 - poisson.pmf(0, mu=mu_h)
                p_away_scores = 1 - poisson.pmf(0, mu=mu_a)
                prob_btts = p_home_scores * p_away_scores
                pred = "yes" if prob_btts >= 0.5 else "no"
                prob = prob_btts if pred == "yes" else 1 - prob_btts
                results.append({
                    "prediction": pred,
                    "probability": float(prob),
                    "prob_yes": float(prob_btts),
                    "prob_no": float(1 - prob_btts),
                    "exp_home": float(mu_h),
                    "exp_away": float(mu_a),
                })
            elif market.startswith("ht_over_"):
                # Use half the expected goals as HT approximation
                line = float(market.split("_")[2])
                mu_ht = mu_total * 0.43  # ~43% of goals in 1st half historically
                prob_over = 1 - poisson.cdf(int(line), mu=mu_ht)
                pred = "yes" if prob_over >= 0.5 else "no"
                prob = prob_over if pred == "yes" else 1 - prob_over
                results.append({
                    "prediction": pred,
                    "probability": float(prob),
                    "prob_yes": float(prob_over),
                    "prob_no": float(1 - prob_over),
                    "exp_ht": float(mu_ht),
                })
            elif market.startswith("2h_over_"):
                line = float(market.split("_")[2])
                mu_2h = mu_total * 0.57  # ~57% of goals in 2nd half
                prob_over = 1 - poisson.cdf(int(line), mu=mu_2h)
                pred = "yes" if prob_over >= 0.5 else "no"
                prob = prob_over if pred == "yes" else 1 - prob_over
                results.append({
                    "prediction": pred,
                    "probability": float(prob),
                    "prob_yes": float(prob_over),
                    "prob_no": float(1 - prob_over),
                    "exp_2h": float(mu_2h),
                })

        return results

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model_home, self.path_home)
        joblib.dump(self.model_away, self.path_away)
        logger.info("Modelos Poisson guardados")

    def load(self):
        if os.path.exists(self.path_home) and os.path.exists(self.path_away):
            self.model_home = joblib.load(self.path_home)
            self.model_away = joblib.load(self.path_away)
            logger.info("Modelos Poisson cargados")
            return True
        return False

    def get_feature_importance(self) -> dict:
        if self.model_home is None:
            return {}
        imp_h = self.model_home.feature_importances_
        imp_a = self.model_away.feature_importances_
        avg = (imp_h + imp_a) / 2
        return {name: float(v) for name, v in zip(self.feature_names, avg)}
