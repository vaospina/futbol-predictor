"""
Entrena todos los modelos de goles y evalúa vs baseline.

Mercados:
  - over_1.5, over_2.5, over_3.5 (goles totales)
  - btts (ambos equipos marcan)
  - ht_over_0.5, ht_over_1.5 (goles primer tiempo)
  - 2h_over_0.5, 2h_over_1.5 (goles segundo tiempo)

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
from models.feature_engineering import build_match_features, MATCH_FEATURE_NAMES
from models.goals_predictor import GoalsPredictor
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


def prepare_data(matches, label_fn, filter_fn=None):
    X_list = []
    y_list = []
    for m in matches:
        if m.get("home_score") is None:
            continue
        if filter_fn and not filter_fn(m):
            continue
        try:
            features = build_match_features(m)
            vec = [features.get(f, 0) for f in MATCH_FEATURE_NAMES]
            label = label_fn(m)
            if label is None:
                continue
            X_list.append(vec)
            y_list.append(label)
        except Exception:
            continue
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# Label functions
def label_over(line):
    def fn(m):
        total = m["home_score"] + m["away_score"]
        return 1 if total > line else 0
    return fn

def label_btts(m):
    return 1 if m["home_score"] > 0 and m["away_score"] > 0 else 0

def label_ht_over(line):
    def fn(m):
        ht = (m.get("home_ht_score") or 0) + (m.get("away_ht_score") or 0)
        return 1 if ht > line else 0
    return fn

def label_2h_over(line):
    def fn(m):
        ht_h = m.get("home_ht_score")
        ht_a = m.get("away_ht_score")
        if ht_h is None or ht_a is None:
            return None
        goals_2h = (m["home_score"] - ht_h) + (m["away_score"] - ht_a)
        return 1 if goals_2h > line else 0
    return fn

def has_ht_scores(m):
    return m.get("home_ht_score") is not None and m.get("away_ht_score") is not None


MARKETS = [
    # (name, label_fn, filter_fn, description)
    ("over_1.5", label_over(1.5), None, "Over 1.5 goles"),
    ("over_2.5", label_over(2.5), None, "Over 2.5 goles"),
    ("over_3.5", label_over(3.5), None, "Over 3.5 goles"),
    ("btts", label_btts, None, "Ambos marcan"),
    ("ht_over_0.5", label_ht_over(0.5), has_ht_scores, "Over 0.5 goles 1T"),
    ("ht_over_1.5", label_ht_over(1.5), has_ht_scores, "Over 1.5 goles 1T"),
    ("2h_over_0.5", label_2h_over(0.5), has_ht_scores, "Over 0.5 goles 2T"),
    ("2h_over_1.5", label_2h_over(1.5), has_ht_scores, "Over 1.5 goles 2T"),
]


def main():
    matches, home_idx, away_idx, all_idx = build_caches()
    install_patches(home_idx, away_idx, all_idx)

    version = datetime.now().strftime("v%Y%m%d_%H%M")
    results = {}

    for market_name, label_fn, filter_fn, desc in MARKETS:
        logger.info(f"=== {desc} ({market_name}) ===")

        X, y = prepare_data(matches, label_fn, filter_fn)
        if len(X) < 100:
            logger.warning(f"  Datos insuficientes: {len(X)}")
            results[market_name] = {"error": "insufficient_data", "samples": len(X)}
            continue

        baseline = float(np.mean(y))
        baseline_acc = max(baseline, 1 - baseline)

        predictor = GoalsPredictor(market_name)
        metrics = predictor.train(X, y)
        metrics["baseline_pct"] = round(baseline * 100, 1)
        metrics["baseline_acc"] = round(baseline_acc * 100, 1)
        metrics["beats_baseline"] = metrics["accuracy_cv"] > baseline_acc

        if metrics["beats_baseline"]:
            predictor.save()
            deactivate_models(f"goals_{market_name}")
            insert_model_version({
                "version": f"{version}_{market_name}",
                "model_type": f"goals_{market_name}",
                "training_samples": metrics["training_samples"],
                "accuracy_cv": metrics["accuracy_cv"],
                "f1_score": metrics["f1_score"],
                "log_loss": metrics["log_loss"],
                "is_active": True,
                "model_binary": None,
                "feature_importance": json.dumps(predictor.get_feature_importance()),
                "notes": f"{desc}: acc={metrics['accuracy_cv']:.3f} vs baseline={metrics['baseline_acc']:.1f}%",
            })
            logger.info(f"  ACTIVADO: {metrics['accuracy_cv']:.1%} > baseline {metrics['baseline_acc']:.1f}%")
        else:
            logger.info(f"  DESHABILITADO: {metrics['accuracy_cv']:.1%} <= baseline {metrics['baseline_acc']:.1f}%")

        results[market_name] = metrics

    logger.info("=== RESUMEN ===")
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
