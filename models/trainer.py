"""
Pipeline completo de entrenamiento de los 3 modelos.
Incluye carga de datos historicos, preparacion de features y entrenamiento.
"""
import numpy as np
import json
from datetime import date, datetime
from db.models import (
    fetch_all, get_config, set_config,
    insert_model_version, deactivate_models, activate_model,
    get_top_shooters_by_league,
)
from models.feature_engineering import (
    build_match_features, build_corners_features, build_player_features,
    MATCH_FEATURE_NAMES, CORNERS_FEATURE_NAMES, PLAYER_FEATURE_NAMES,
)
from models.match_predictor import MatchPredictor
from models.corners_predictor import CornersPredictor
from models.shots_predictor import ShotsPredictor
from config.leagues import ALL_LEAGUES
from config.settings import CURRENT_SEASON
from utils.logger import get_logger

logger = get_logger(__name__)


def prepare_training_data_1x2():
    """Prepara datos de entrenamiento para el modelo 1X2."""
    matches = fetch_all(
        """SELECT * FROM matches
        WHERE status = 'finished' AND home_score IS NOT NULL
        ORDER BY match_date ASC"""
    )
    logger.info(f"Preparando datos 1X2 con {len(matches)} partidos")

    X_list = []
    y_list = []

    for match in matches:
        try:
            features = build_match_features(match)
            feature_vector = [features.get(f, 0) for f in MATCH_FEATURE_NAMES]

            hs = match["home_score"]
            aws = match["away_score"]
            if hs > aws:
                label = "home_win"
            elif hs == aws:
                label = "draw"
            else:
                label = "away_win"

            X_list.append(feature_vector)
            y_list.append(label)
        except Exception as e:
            logger.debug(f"Error procesando match {match.get('id')}: {e}")

    logger.info(f"Datos 1X2 preparados: {len(X_list)} muestras")
    return np.array(X_list, dtype=np.float32), np.array(y_list)


def prepare_training_data_corners():
    """Prepara datos de entrenamiento para el modelo de corners."""
    matches = fetch_all(
        """SELECT * FROM matches
        WHERE status = 'finished' AND home_corners IS NOT NULL AND away_corners IS NOT NULL
        ORDER BY match_date ASC"""
    )
    logger.info(f"Preparando datos corners con {len(matches)} partidos")

    X_list = []
    y_list = []

    for match in matches:
        try:
            all_features = build_match_features(match)
            corner_features = build_corners_features(all_features)
            feature_vector = [corner_features.get(f, 0) for f in CORNERS_FEATURE_NAMES]

            total_corners = match["home_corners"] + match["away_corners"]

            X_list.append(feature_vector)
            y_list.append(total_corners)
        except Exception as e:
            logger.debug(f"Error procesando corners match {match.get('id')}: {e}")

    logger.info(f"Datos corners preparados: {len(X_list)} muestras")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def prepare_training_data_shots():
    """Prepara datos de entrenamiento para el modelo de shots.

    Usa SOLO partidos anteriores a la fecha del partido (sin leakage de
    stats del mismo partido ni de partidos futuros).
    """
    from datetime import date as _date
    players = fetch_all(
        """SELECT ps.*, m.home_team_id, m.away_team_id, m.home_team, m.away_team,
                  m.league_id
        FROM player_stats ps
        JOIN matches m ON ps.match_id = m.id
        WHERE ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
        ORDER BY ps.match_date ASC"""
    )
    logger.info(f"Preparando datos shots con {len(players)} registros")

    X_list = []
    y_list = []

    for p in players:
        try:
            match_date = p.get("match_date")
            if isinstance(match_date, str):
                match_date = _date.fromisoformat(match_date[:10])
            elif hasattr(match_date, "date"):
                match_date = match_date.date()

            # Features del equipo calculadas SOLO con partidos previos
            mock_match = {
                "home_team_id": p.get("home_team_id"),
                "away_team_id": p.get("away_team_id"),
                "home_team": p.get("home_team"),
                "away_team": p.get("away_team"),
                "match_date": match_date,
                "id": p.get("match_id"),
            }
            team_feats = build_match_features(mock_match)

            player_features = build_player_features(
                p["player_id"], team_feats, before_date=match_date
            )
            if player_features is None:
                continue

            feature_vector = [player_features.get(f, 0) for f in PLAYER_FEATURE_NAMES]
            X_list.append(feature_vector)
            y_list.append(p["shots_on_target"])
        except Exception as e:
            logger.debug(f"Error procesando shots player {p.get('player_id')}: {e}")

    logger.info(f"Datos shots preparados: {len(X_list)} muestras")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_all_models(version_suffix: str = None) -> dict:
    """
    Entrena los 3 modelos y guarda versiones.
    Retorna metricas de cada modelo.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    version = version_suffix or f"v{timestamp}"

    results = {}

    # --- Modelo 1X2 ---
    logger.info("=== Entrenando Modelo 1X2 ===")
    X_1x2, y_1x2 = prepare_training_data_1x2()
    if len(X_1x2) >= 30:
        predictor_1x2 = MatchPredictor()
        metrics_1x2 = predictor_1x2.train(X_1x2, y_1x2)
        predictor_1x2.save()

        deactivate_models("1x2")
        insert_model_version({
            "version": f"{version}_1x2",
            "model_type": "1x2",
            "training_samples": metrics_1x2["training_samples"],
            "accuracy_cv": metrics_1x2["accuracy_cv"],
            "f1_score": metrics_1x2["f1_score"],
            "log_loss": metrics_1x2["log_loss"],
            "is_active": True,
            "model_binary": None,
            "feature_importance": json.dumps(predictor_1x2.get_feature_importance()),
            "notes": f"Entrenado con {metrics_1x2['training_samples']} muestras",
        })
        set_config("model_active_1x2", f"{version}_1x2")
        results["1x2"] = metrics_1x2
    else:
        logger.warning(f"Insuficientes datos para 1X2: {len(X_1x2)} (minimo 30)")
        results["1x2"] = {"error": "insufficient_data", "samples": len(X_1x2)}

    # --- Modelo Corners ---
    logger.info("=== Entrenando Modelo Corners ===")
    X_corners, y_corners = prepare_training_data_corners()
    if len(X_corners) >= 30:
        predictor_corners = CornersPredictor()
        metrics_corners = predictor_corners.train(X_corners, y_corners)
        predictor_corners.save()

        deactivate_models("corners")
        insert_model_version({
            "version": f"{version}_corners",
            "model_type": "corners",
            "training_samples": metrics_corners["training_samples"],
            "accuracy_cv": metrics_corners["r2_cv"],
            "f1_score": None,
            "log_loss": metrics_corners["mae_cv"],
            "is_active": True,
            "model_binary": None,
            "feature_importance": json.dumps(predictor_corners.get_feature_importance()),
            "notes": f"R2={metrics_corners['r2_cv']:.3f}, MAE={metrics_corners['mae_cv']:.2f}",
        })
        set_config("model_active_corners", f"{version}_corners")
        results["corners"] = metrics_corners
    else:
        logger.warning(f"Insuficientes datos para corners: {len(X_corners)}")
        results["corners"] = {"error": "insufficient_data", "samples": len(X_corners)}

    # --- Modelo Shots ---
    logger.info("=== Entrenando Modelo Shots ===")
    X_shots, y_shots = prepare_training_data_shots()
    if len(X_shots) >= 30:
        predictor_shots = ShotsPredictor()
        metrics_shots = predictor_shots.train(X_shots, y_shots)
        predictor_shots.save()

        deactivate_models("player_shots")
        insert_model_version({
            "version": f"{version}_shots",
            "model_type": "player_shots",
            "training_samples": metrics_shots["training_samples"],
            "accuracy_cv": metrics_shots["r2_cv"],
            "f1_score": None,
            "log_loss": metrics_shots["mae_cv"],
            "is_active": True,
            "model_binary": None,
            "feature_importance": json.dumps(predictor_shots.get_feature_importance()),
            "notes": f"R2={metrics_shots['r2_cv']:.3f}, MAE={metrics_shots['mae_cv']:.2f}",
        })
        set_config("model_active_shots", f"{version}_shots")
        results["shots"] = metrics_shots
    else:
        logger.warning(f"Insuficientes datos para shots: {len(X_shots)}")
        results["shots"] = {"error": "insufficient_data", "samples": len(X_shots)}

    logger.info(f"Entrenamiento completado. Version: {version}")
    return results
