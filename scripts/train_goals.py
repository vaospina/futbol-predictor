"""
Entrena modelos de goles: XGBoost clasificador vs Poisson regresión.
Evalúa ambos contra baseline y activa solo los que superen.

Uso:
    python -m scripts.train_goals
"""
import json
import numpy as np
from datetime import date, datetime
from collections import defaultdict

from db.models import fetch_all
from db import models as db_models
from models import feature_engineering as fe
from models.feature_engineering import build_match_features, GOALS_FEATURE_NAMES
from models.goals_predictor import GoalsPredictor
from models.goals_poisson import GoalsPoissonPredictor
from db.models import insert_model_version, deactivate_models, set_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _to_date(d):
    if d is None:
        return None
    if isinstance(d, str):
        return date.fromisoformat(d[:10])
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return None


def build_caches():
    logger.info("Cargando matches en memoria...")
    matches = fetch_all(
        "SELECT * FROM matches WHERE status='finished' ORDER BY match_date ASC"
    )
    logger.info(f"  {len(matches)} matches cargados")

    for m in matches:
        m["_date"] = _to_date(m.get("match_date"))

    home_idx = defaultdict(list)
    away_idx = defaultdict(list)
    all_idx = defaultdict(list)

    for m in matches:
        h = m.get("home_team_id")
        a = m.get("away_team_id")
        if h:
            home_idx[h].append(m)
            all_idx[h].append(m)
        if a:
            away_idx[a].append(m)
            all_idx[a].append(m)

    return matches, home_idx, away_idx, all_idx


def install_patches(home_idx, away_idx, all_idx):
    def get_team_home_matches(team_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = home_idx.get(team_id, [])
        filtered = [m for m in lst if m["_date"] and m["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_team_away_matches(team_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = away_idx.get(team_id, [])
        filtered = [m for m in lst if m["_date"] and m["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_team_last_matches(team_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = all_idx.get(team_id, [])
        filtered = [m for m in lst if m["_date"] and m["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_h2h_matches(t1, t2, n=5, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = all_idx.get(t1, [])
        both = [
            m for m in lst
            if m["_date"] and m["_date"] < before
            and (
                (m.get("home_team_id") == t1 and m.get("away_team_id") == t2)
                or (m.get("home_team_id") == t2 and m.get("away_team_id") == t1)
            )
        ]
        return list(reversed(both[-n:]))

    def get_match_odds_summary(match_id):
        return {}

    db_models.get_team_home_matches = get_team_home_matches
    db_models.get_team_away_matches = get_team_away_matches
    db_models.get_team_last_matches = get_team_last_matches
    db_models.get_h2h_matches = get_h2h_matches
    fe.get_team_home_matches = get_team_home_matches
    fe.get_team_away_matches = get_team_away_matches
    fe.get_team_last_matches = get_team_last_matches
    fe.get_h2h_matches = get_h2h_matches
    fe.get_match_odds_summary = get_match_odds_summary


def prepare_goals_data(matches, filter_fn=None):
    """Build feature matrix + goal targets for all matches."""
    X_list = []
    y_home_list = []
    y_away_list = []
    meta = []  # keep match data for label functions

    for m in matches:
        if m.get("home_score") is None:
            continue
        if filter_fn and not filter_fn(m):
            continue
        try:
            features = build_match_features(m)
            vec = [features.get(f, 0) for f in GOALS_FEATURE_NAMES]
            X_list.append(vec)
            y_home_list.append(m["home_score"])
            y_away_list.append(m["away_score"])
            meta.append(m)
        except Exception:
            continue

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_home_list, dtype=np.float32),
        np.array(y_away_list, dtype=np.float32),
        meta,
    )


def has_ht_scores(m):
    return m.get("home_ht_score") is not None and m.get("away_ht_score") is not None


# Market definitions: (name, label_fn on match, description)
def get_label(match, market):
    hs, aws = match["home_score"], match["away_score"]
    total = hs + aws

    if market == "over_1.5":
        return 1 if total > 1.5 else 0
    elif market == "over_2.5":
        return 1 if total > 2.5 else 0
    elif market == "over_3.5":
        return 1 if total > 3.5 else 0
    elif market == "btts":
        return 1 if hs > 0 and aws > 0 else 0
    elif market == "ht_over_0.5":
        ht = (match.get("home_ht_score") or 0) + (match.get("away_ht_score") or 0)
        return 1 if ht > 0.5 else 0
    elif market == "ht_over_1.5":
        ht = (match.get("home_ht_score") or 0) + (match.get("away_ht_score") or 0)
        return 1 if ht > 1.5 else 0
    elif market == "2h_over_0.5":
        hth = match.get("home_ht_score")
        hta = match.get("away_ht_score")
        if hth is None:
            return None
        g2h = (hs - hth) + (aws - hta)
        return 1 if g2h > 0.5 else 0
    elif market == "2h_over_1.5":
        hth = match.get("home_ht_score")
        hta = match.get("away_ht_score")
        if hth is None:
            return None
        g2h = (hs - hth) + (aws - hta)
        return 1 if g2h > 1.5 else 0
    return None


MARKETS_FULLTIME = ["over_1.5", "over_2.5", "over_3.5", "btts"]
MARKETS_HALFTIME = ["ht_over_0.5", "ht_over_1.5", "2h_over_0.5", "2h_over_1.5"]
ALL_MARKETS = MARKETS_FULLTIME + MARKETS_HALFTIME


def evaluate_xgb(X, meta, market):
    """Train XGBoost classifier for a market, return metrics."""
    y = np.array([get_label(m, market) for m in meta], dtype=np.int32)
    valid = y >= 0  # filter None labels
    X_valid = X[valid]
    y_valid = y[valid]

    if len(X_valid) < 100:
        return None

    baseline = float(np.mean(y_valid))
    baseline_acc = max(baseline, 1 - baseline)

    predictor = GoalsPredictor(market)
    predictor.feature_names = GOALS_FEATURE_NAMES
    metrics = predictor.train(X_valid, y_valid)
    metrics["baseline_pct"] = round(baseline * 100, 1)
    metrics["baseline_acc"] = round(baseline_acc * 100, 1)
    metrics["beats_baseline"] = metrics["accuracy_cv"] > baseline_acc
    metrics["predictor"] = predictor
    return metrics


def evaluate_poisson(poisson_model, X, meta, market):
    """Evaluate Poisson model on a market via time-series simulation."""
    from sklearn.model_selection import TimeSeriesSplit

    y = np.array([get_label(m, market) for m in meta], dtype=np.int32)
    valid = y >= 0
    X_valid = X[valid]
    y_valid = y[valid]

    if len(X_valid) < 100:
        return None

    baseline = float(np.mean(y_valid))
    baseline_acc = max(baseline, 1 - baseline)

    # Evaluate by predicting on the full set (model already trained)
    preds = poisson_model.predict_market(X_valid, market)
    y_pred = np.array([1 if p["prob_yes"] >= 0.5 else 0 for p in preds])
    accuracy = float(np.mean(y_pred == y_valid))

    # Also do proper CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accs = []
    y_home_all = np.array([m["home_score"] for m in meta], dtype=np.float32)[valid]
    y_away_all = np.array([m["away_score"] for m in meta], dtype=np.float32)[valid]

    for train_idx, test_idx in tscv.split(X_valid):
        X_tr, X_te = X_valid[train_idx], X_valid[test_idx]
        yh_tr, ya_tr = y_home_all[train_idx], y_away_all[train_idx]
        y_te = y_valid[test_idx]

        fold_model = GoalsPoissonPredictor()
        fold_model.train(X_tr, yh_tr, ya_tr)
        fold_preds = fold_model.predict_market(X_te, market)
        fold_y_pred = np.array([1 if p["prob_yes"] >= 0.5 else 0 for p in fold_preds])
        cv_accs.append(float(np.mean(fold_y_pred == y_te)))

    return {
        "accuracy_cv": float(np.mean(cv_accs)),
        "accuracy_full": accuracy,
        "baseline_pct": round(baseline * 100, 1),
        "baseline_acc": round(baseline_acc * 100, 1),
        "beats_baseline": float(np.mean(cv_accs)) > baseline_acc,
        "training_samples": len(X_valid),
    }


def main():
    matches, home_idx, away_idx, all_idx = build_caches()
    install_patches(home_idx, away_idx, all_idx)

    version = datetime.now().strftime("v%Y%m%d_%H%M")

    # Prepare data — fulltime (all matches) and halftime (only those with HT)
    logger.info("Preparando datos fulltime...")
    X_ft, yh_ft, ya_ft, meta_ft = prepare_goals_data(matches)
    logger.info(f"  {len(X_ft)} muestras fulltime")

    logger.info("Preparando datos halftime...")
    X_ht, yh_ht, ya_ht, meta_ht = prepare_goals_data(matches, filter_fn=has_ht_scores)
    logger.info(f"  {len(X_ht)} muestras halftime")

    # Train Poisson models
    logger.info("=== Entrenando modelo Poisson (fulltime) ===")
    poisson_ft = GoalsPoissonPredictor()
    poisson_metrics_ft = poisson_ft.train(X_ft, yh_ft, ya_ft)

    logger.info("=== Entrenando modelo Poisson (halftime) ===")
    poisson_ht = GoalsPoissonPredictor()
    poisson_metrics_ht = poisson_ht.train(X_ht, yh_ht, ya_ht)

    # Compare all markets
    comparison = {}
    activated = []

    for market in ALL_MARKETS:
        is_ht = market in MARKETS_HALFTIME
        X = X_ht if is_ht else X_ft
        meta = meta_ht if is_ht else meta_ft
        poisson_model = poisson_ht if is_ht else poisson_ft

        logger.info(f"=== {market} ===")

        xgb_result = evaluate_xgb(X, meta, market)
        poi_result = evaluate_poisson(poisson_model, X, meta, market)

        xgb_acc = xgb_result["accuracy_cv"] if xgb_result else 0
        poi_acc = poi_result["accuracy_cv"] if poi_result else 0
        baseline_acc = xgb_result["baseline_acc"] if xgb_result else 0

        best_method = "poisson" if poi_acc > xgb_acc else "xgboost"
        best_acc = max(xgb_acc, poi_acc)
        beats = best_acc > baseline_acc / 100

        comparison[market] = {
            "xgb_acc": round(xgb_acc * 100, 1),
            "poisson_acc": round(poi_acc * 100, 1),
            "baseline_acc": baseline_acc,
            "best_method": best_method,
            "best_acc": round(best_acc * 100, 1),
            "beats_baseline": beats,
            "samples": xgb_result["training_samples"] if xgb_result else 0,
        }

        if beats:
            logger.info(
                f"  ACTIVADO ({best_method}): {best_acc:.1%} > baseline {baseline_acc}%"
            )
            activated.append(market)

            if best_method == "poisson":
                poisson_model.save()
                deactivate_models(f"goals_{market}")
                insert_model_version({
                    "version": f"{version}_{market}_poisson",
                    "model_type": f"goals_{market}",
                    "training_samples": poi_result["training_samples"],
                    "accuracy_cv": poi_result["accuracy_cv"],
                    "f1_score": None,
                    "log_loss": None,
                    "is_active": True,
                    "model_binary": None,
                    "feature_importance": json.dumps(poisson_model.get_feature_importance()),
                    "notes": f"Poisson: {poi_result['accuracy_cv']:.3f} vs baseline {baseline_acc}%",
                })
            else:
                predictor = xgb_result["predictor"]
                predictor.save()
                deactivate_models(f"goals_{market}")
                insert_model_version({
                    "version": f"{version}_{market}_xgb",
                    "model_type": f"goals_{market}",
                    "training_samples": xgb_result["training_samples"],
                    "accuracy_cv": xgb_result["accuracy_cv"],
                    "f1_score": xgb_result.get("f1_score"),
                    "log_loss": xgb_result.get("log_loss"),
                    "is_active": True,
                    "model_binary": None,
                    "feature_importance": json.dumps(predictor.get_feature_importance()),
                    "notes": f"XGBoost: {xgb_result['accuracy_cv']:.3f} vs baseline {baseline_acc}%",
                })
        else:
            logger.info(
                f"  DESHABILITADO: best={best_acc:.1%} <= baseline {baseline_acc}%"
            )

    logger.info(f"\n=== MERCADOS ACTIVADOS: {activated if activated else 'NINGUNO'} ===")
    print(json.dumps(comparison, indent=2))
    return comparison, activated


if __name__ == "__main__":
    main()
