"""
Flujo de la manana (6:00 AM hora Colombia / 11:00 UTC).
Genera predicciones para los partidos del dia.
"""
import numpy as np
from datetime import date
from data.api_football import ApiFootballClient, parse_fixture, parse_fixture_statistics
from data.news_collector import fetch_team_news
from data.sentiment import analyze_team_news
from data.odds_collector import collect_odds_for_fixture
from models.feature_engineering import (
    build_match_features, build_corners_features, build_player_features,
    MATCH_FEATURE_NAMES, CORNERS_FEATURE_NAMES, PLAYER_FEATURE_NAMES,
)
from models.match_predictor import MatchPredictor
from models.corners_predictor import CornersPredictor
from models.shots_predictor import ShotsPredictor
from models.expected_value import expected_value
from models.evaluator import get_current_threshold, get_model_version
from db.models import (
    upsert_match, insert_prediction, get_cumulative_stats,
    get_current_streak, get_worst_streak, get_config,
    get_top_shooters_by_league, get_match_by_fixture_id,
)
from notifications.telegram import send_prediction_message, send_telegram
from config.leagues import ALL_LEAGUES
from config.settings import MAX_DAILY_PREDICTIONS, CURRENT_SEASON, ENABLE_CORNERS_MODEL
from utils.helpers import today_colombia
from utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_predictions():
    """Flujo completo de predicciones diarias."""
    logger.info("=== INICIANDO FLUJO DE PREDICCIONES ===")

    try:
        api = ApiFootballClient()
        today = today_colombia()
        threshold = get_current_threshold()

        # 1. Cargar modelos
        match_predictor = MatchPredictor()
        corners_predictor = CornersPredictor()
        shots_predictor = ShotsPredictor()

        match_predictor.load()
        corners_predictor.load()
        shots_predictor.load()

        # 2. Obtener partidos del dia
        all_fixtures = []
        for league_id, league_info in ALL_LEAGUES.items():
            fixtures = api.get_fixtures_by_date(league_id, today)
            for f in fixtures:
                f["_league_info"] = league_info
            all_fixtures.extend(fixtures)
            logger.info(f"{league_info['name']}: {len(fixtures)} partidos hoy")

        if not all_fixtures:
            logger.info("No hay partidos hoy en las ligas configuradas")
            send_telegram(f"\U0001f3df *Sin partidos hoy* ({today.strftime('%d/%m/%Y')})\nNo hay partidos programados en las ligas monitoreadas.")
            return

        logger.info(f"Total partidos hoy: {len(all_fixtures)}")

        # 3. Procesar cada partido y generar candidatos
        all_candidates = []

        for fixture_data in all_fixtures:
            try:
                match_data = parse_fixture(fixture_data)

                # Guardar en BD
                match_id = upsert_match(match_data)
                if not match_id:
                    continue

                fixture_id = match_data["api_fixture_id"]
                home_team = match_data["home_team"]
                away_team = match_data["away_team"]
                home_id = match_data["home_team_id"]
                away_id = match_data["away_team_id"]

                logger.info(f"Procesando: {home_team} vs {away_team}")

                # Recolectar odds
                collect_odds_for_fixture(fixture_id, match_id)

                # Analizar noticias
                home_news = fetch_team_news(home_team)
                away_news = fetch_team_news(away_team)
                sentiment_home = analyze_team_news(home_id, home_team, home_news)
                sentiment_away = analyze_team_news(away_id, away_team, away_news)

                # Obtener posiciones de la tabla
                match_data["id"] = match_id
                match_data["home_league_position"] = 10  # default
                match_data["away_league_position"] = 10

                standings = api.get_standings(match_data["league_id"])
                for team_standing in standings:
                    team = team_standing.get("team", {})
                    if team.get("id") == home_id:
                        match_data["home_league_position"] = team_standing.get("rank", 10)
                    elif team.get("id") == away_id:
                        match_data["away_league_position"] = team_standing.get("rank", 10)

                # Construir features
                features = build_match_features(match_data, sentiment_home, sentiment_away)

                # --- Prediccion 1X2 ---
                if match_predictor.model:
                    X_1x2 = np.array([[features.get(f, 0) for f in MATCH_FEATURE_NAMES]], dtype=np.float32)
                    preds_1x2 = match_predictor.predict(X_1x2)
                    if preds_1x2:
                        pred = preds_1x2[0]
                        odds_key = {
                            "home_win": "odds_home_win",
                            "draw": "odds_draw",
                            "away_win": "odds_away_win",
                        }.get(pred["prediction"], "odds_home_win")
                        odds = features.get(odds_key, 2.0)
                        ev = expected_value(pred["probability"], odds)

                        prediction_label = {
                            "home_win": f"{home_team} gana",
                            "draw": "Empate",
                            "away_win": f"{away_team} gana",
                        }.get(pred["prediction"], pred["prediction"])

                        all_candidates.append({
                            "match_id": match_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "market_type": "1x2",
                            "prediction": prediction_label,
                            "raw_prediction": pred["prediction"],
                            "probability": pred["probability"],
                            "odds": odds,
                            "expected_value": ev,
                        })

                # --- Prediccion Corners ---
                if corners_predictor.model and ENABLE_CORNERS_MODEL:
                    corner_features = build_corners_features(features)
                    X_corners = np.array([[corner_features.get(f, 0) for f in CORNERS_FEATURE_NAMES]], dtype=np.float32)
                    preds_corners = corners_predictor.predict(X_corners, [9.5])
                    if preds_corners:
                        pred = preds_corners[0]
                        odds_corners = features.get("odds_over_corners", 1.9)
                        ev = expected_value(pred["probability"], odds_corners)

                        all_candidates.append({
                            "match_id": match_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "market_type": "corners",
                            "prediction": f"+{pred['line']} corners" if "over" in pred["prediction"] else f"-{pred['line']} corners",
                            "raw_prediction": pred["prediction"],
                            "probability": pred["probability"],
                            "odds": odds_corners,
                            "expected_value": ev,
                        })

                # --- Prediccion Shots (top shooters) ---
                if shots_predictor.model:
                    league_id = match_data["league_id"]
                    top_shooters = get_top_shooters_by_league(league_id, CURRENT_SEASON, 3)
                    for shooter in top_shooters:
                        if shooter["team_name"] not in (home_team, away_team):
                            continue
                        player_feats = build_player_features(shooter["player_id"], features, before_date=today)
                        if player_feats is None:
                            continue

                        X_shots = np.array([[player_feats.get(f, 0) for f in PLAYER_FEATURE_NAMES]], dtype=np.float32)
                        preds_shots = shots_predictor.predict(X_shots, [1.5])
                        if preds_shots:
                            pred = preds_shots[0]
                            ev = expected_value(pred["probability"], 1.35)

                            all_candidates.append({
                                "match_id": match_id,
                                "home_team": home_team,
                                "away_team": away_team,
                                "market_type": "player_shots",
                                "prediction": f"{shooter['player_name']} +1.5 disparos a puerta",
                                "raw_prediction": pred["prediction"],
                                "probability": pred["probability"],
                                "odds": 1.35,
                                "expected_value": ev,
                                "player_name": shooter["player_name"],
                            })

            except Exception as e:
                logger.error(f"Error procesando fixture: {e}")
                continue

        # 4. Filtrar por umbral y EV positivo
        filtered = [
            c for c in all_candidates
            if c["probability"] >= threshold and c["expected_value"] > 0
        ]

        # 5. Ordenar por probabilidad y tomar top N
        filtered.sort(key=lambda x: x["probability"], reverse=True)
        selected = filtered[:MAX_DAILY_PREDICTIONS]

        logger.info(
            f"Candidatos: {len(all_candidates)} | "
            f"Filtrados (umbral={threshold:.0%}, EV>0): {len(filtered)} | "
            f"Seleccionados: {len(selected)}"
        )

        # 6. Guardar en BD
        model_version = get_model_version("1x2")
        for pred in selected:
            insert_prediction({
                "match_id": pred["match_id"],
                "prediction_date": today,
                "market_type": pred["market_type"],
                "prediction": pred["prediction"],
                "probability": pred["probability"],
                "odds": pred["odds"],
                "expected_value": pred["expected_value"],
                "model_version": model_version,
            })

        # 7. Enviar Telegram
        if selected:
            cum_stats = get_cumulative_stats()
            streak = get_current_streak()
            worst = get_worst_streak()
            send_prediction_message(
                selected, model_version, threshold,
                cum_stats, streak, worst
            )
        else:
            send_telegram(
                f"\U0001f3df *Predicciones {today.strftime('%d/%m/%Y')}*\n\n"
                f"No hay predicciones que superen el umbral ({int(threshold * 100)}%) "
                f"con EV positivo hoy."
            )

        logger.info("=== FLUJO DE PREDICCIONES COMPLETADO ===")

    except Exception as e:
        logger.error(f"Error en flujo de predicciones: {e}")
        send_telegram(f"\u26a0\ufe0f Error en predicciones: {str(e)[:200]}")
