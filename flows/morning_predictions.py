"""
Flujo de la manana (6:00 AM hora Colombia / 11:00 UTC).
Genera predicciones para los partidos del dia.

Shots en 2 fases:
  Fase 1 (6 AM): predice shots usando historial de titularidades (~60% ultimos 5)
                 guarda como 'preliminary_shots' en BD
  Fase 2 (kickoff-60min): job dinamico verifica lineup confirmado y confirma/cancela
"""
import pytz
import numpy as np
from datetime import date, datetime, timedelta
from data.api_football import ApiFootballClient, parse_fixture, parse_fixture_statistics, parse_lineups
from data.news_collector import fetch_team_news
from data.sentiment import analyze_team_news
from data.odds_collector import collect_odds_for_fixture, get_match_odds_summary
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
    get_player_recent_stats,
)
from notifications.telegram import send_prediction_message, send_telegram
from config.leagues import ALL_LEAGUES
from config.settings import CURRENT_SEASON, ENABLE_CORNERS_MODEL
from utils.helpers import today_colombia
from utils import scheduler_registry
from utils.logger import get_logger

logger = get_logger(__name__)


def _is_likely_starter(recent_stats: list, min_fraction: float = 0.6) -> bool:
    """True si el jugador titularizó (60+ min) en >= 60% de sus últimas apariciones."""
    if not recent_stats:
        return False
    started = sum(1 for s in recent_stats if (s.get("minutes_played") or 0) >= 60)
    return (started / len(recent_stats)) >= min_fraction


def _schedule_lineup_checks(saved_predictions: list):
    """Para cada shot preliminar guardado, programa un job de verificación de lineup."""
    from flows.lineup_check import verify_shots_for_match

    scheduler = scheduler_registry.get_scheduler()
    if not scheduler:
        logger.warning("Scheduler no registrado — no se pueden programar lineup checks")
        return

    now_utc = datetime.now(pytz.UTC)
    scheduled_fixtures = set()

    for pred in saved_predictions:
        if pred.get("data_source") != "preliminary_shots":
            continue
        fixture_id = pred.get("fixture_id")
        match_id = pred.get("match_id")
        match_date_utc = pred.get("match_date_utc")
        if not fixture_id or not match_date_utc or fixture_id in scheduled_fixtures:
            continue

        try:
            utc_kickoff = datetime.fromisoformat(str(match_date_utc).replace("Z", "+00:00"))
            if utc_kickoff.tzinfo is None:
                utc_kickoff = pytz.UTC.localize(utc_kickoff)
            job_time = utc_kickoff - timedelta(minutes=60)

            if job_time <= now_utc:
                logger.info(
                    f"Kickoff en <60min, verificando lineup ahora: "
                    f"{pred['home_team']} vs {pred['away_team']}"
                )
                verify_shots_for_match(
                    fixture_id, match_id, pred["home_team"], pred["away_team"]
                )
            else:
                scheduler.add_job(
                    verify_shots_for_match,
                    "date",
                    run_date=job_time,
                    args=[fixture_id, match_id, pred["home_team"], pred["away_team"]],
                    id=f"lineup_check_{fixture_id}",
                    replace_existing=True,
                )
                logger.info(
                    f"Lineup check programado: {job_time.strftime('%H:%M')} UTC — "
                    f"{pred['home_team']} vs {pred['away_team']}"
                )
            scheduled_fixtures.add(fixture_id)
        except Exception as e:
            logger.error(f"Error programando lineup check fixture {fixture_id}: {e}")


def run_daily_predictions():
    """Flujo completo de predicciones diarias."""
    logger.info("=== INICIANDO FLUJO DE PREDICCIONES ===")

    try:
        api = ApiFootballClient()
        today = today_colombia()
        threshold = get_current_threshold()

        # Verificar que la API responde antes de generar predicciones
        api_status = api.check_status()
        if not api_status["ok"]:
            if api_status.get("quota_exhausted"):
                logger.warning("Cuota diaria de API-Football agotada, abortando predicciones")
                # Telegram ya fue enviado dentro de check_status()
            else:
                logger.error("API-Football no responde, abortando predicciones")
                send_telegram(
                    "ALERTA: API-Football no responde. "
                    "No se generaron predicciones hoy. "
                    "Verificar estado del servicio.",
                    parse_mode=None,
                )
            return

        reqs = api_status.get("requests", {})
        logger.info(
            f"API-Football OK: {reqs.get('current', '?')}/{reqs.get('limit_day', '?')} requests"
        )

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
            season = league_info.get("season", CURRENT_SEASON)
            fixtures = api.get_fixtures_by_date(league_id, today, season=season)
            for f in fixtures:
                f["_league_info"] = league_info
            all_fixtures.extend(fixtures)
            logger.info(f"{league_info['name']}: {len(fixtures)} partidos hoy")

        if not api.api_healthy:
            logger.error("API falló durante la obtención de fixtures, abortando")
            send_telegram(
                "ALERTA: API-Football falló al obtener partidos. "
                "No se generaron predicciones hoy.",
                parse_mode=None,
            )
            return

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
                league_name = match_data.get("league_name", "")

                logger.info(f"Procesando: {home_team} vs {away_team}")

                # Recolectar odds
                live_odds = collect_odds_for_fixture(api, fixture_id, match_id)

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

                # Construir features (con odds reales si disponibles)
                features = build_match_features(match_data, sentiment_home, sentiment_away)

                # Sobreescribir odds con datos reales del bookmaker
                if live_odds:
                    for key in ("odds_home_win", "odds_draw", "odds_away_win",
                                "odds_over25", "odds_under25", "odds_btts_yes", "odds_btts_no",
                                "odds_over15", "odds_under15", "odds_over35", "odds_under35"):
                        if live_odds.get(key):
                            features[key] = live_odds[key]
                    # Recalcular prob implícita con odds reales
                    hw = features.get("odds_home_win", 2.0)
                    dw = features.get("odds_draw", 3.3)
                    aw = features.get("odds_away_win", 3.5)
                    tp = 1/hw + 1/dw + 1/aw
                    features["odds_implied_prob_home"] = (1/hw) / tp if tp > 0 else 0.33

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
                            "league_name": league_name,
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
                            "league_name": league_name,
                            "market_type": "corners",
                            "prediction": f"+{pred['line']} corners" if "over" in pred["prediction"] else f"-{pred['line']} corners",
                            "raw_prediction": pred["prediction"],
                            "probability": pred["probability"],
                            "odds": odds_corners,
                            "expected_value": ev,
                        })

                # --- Prediccion Shots — Fase 1: historial de titularidades ---
                # Los lineups a las 6 AM casi nunca están disponibles.
                # Usamos frecuencia histórica de titular como proxy.
                # La Fase 2 (job dinámico a kickoff-60min) verificará el lineup real.
                if shots_predictor.model:
                    league_id = match_data["league_id"]
                    # Usar temporada propia de la liga (LATAM=2026, Europa=2025)
                    league_season = fixture_data.get("_league_info", {}).get("season", CURRENT_SEASON)
                    top_shooters = get_top_shooters_by_league(league_id, league_season, 3)

                    relevant_shooters = [
                        s for s in top_shooters
                        if s["team_name"] in (home_team, away_team)
                    ]

                    for shooter in relevant_shooters:
                        pid = shooter["player_id"]

                        # Filtro: titular en >= 60% de últimos 5 partidos
                        recent_stats = get_player_recent_stats(pid, 5, before_date=today)
                        if not _is_likely_starter(recent_stats):
                            logger.info(
                                f"  {shooter['player_name']}: historial titular <60%, descartado"
                            )
                            continue

                        # Guard 1: avg_SOT reciente mínimo para línea "over 1.5"
                        # Con mu < 1.0, P(X >= 2) < 26% — no puede ser candidato viable
                        recent_sot_vals = [s.get("shots_on_target") or 0 for s in recent_stats]
                        avg_sot_recent = sum(recent_sot_vals) / len(recent_sot_vals) if recent_sot_vals else 0
                        hist_over15_rate = (
                            sum(1 for v in recent_sot_vals if v >= 2) / len(recent_sot_vals)
                            if recent_sot_vals else 0
                        )
                        if avg_sot_recent < 1.0:
                            logger.info(
                                f"  {shooter['player_name']}: avg_SOT={avg_sot_recent:.2f} < 1.0 "
                                f"(over15_hist={hist_over15_rate:.0%}) — descartado"
                            )
                            continue

                        is_home = 1 if shooter["team_name"] == home_team else 0
                        features["is_home"] = is_home
                        player_feats = build_player_features(pid, features, before_date=today)
                        if player_feats is None:
                            continue

                        X_shots = np.array([[player_feats.get(f, 0) for f in PLAYER_FEATURE_NAMES]], dtype=np.float32)
                        preds_shots = shots_predictor.predict(X_shots, [1.5])
                        if preds_shots:
                            pred = preds_shots[0]

                            # Guard 2: detección de miscalibración
                            # Si el modelo predice >65% pero el histórico muestra <30% de over_1.5,
                            # es señal de sobreestimación (modelo europeo aplicado a liga LATAM).
                            if pred["probability"] > 0.65 and hist_over15_rate < 0.30:
                                logger.warning(
                                    f"  {shooter['player_name']}: modelo={pred['probability']:.0%} >> "
                                    f"histórico={hist_over15_rate:.0%} — miscalibración, descartado"
                                )
                                continue

                            shots_odds = 1.75
                            ev = expected_value(pred["probability"], shots_odds)

                            all_candidates.append({
                                "match_id": match_id,
                                "home_team": home_team,
                                "away_team": away_team,
                                "league_name": league_name,
                                "market_type": "player_shots",
                                "prediction": f"{shooter['player_name']} +1.5 disparos a puerta",
                                "raw_prediction": pred["prediction"],
                                "probability": pred["probability"],
                                "odds": shots_odds,
                                "expected_value": ev,
                                "player_name": shooter["player_name"],
                                "player_id": pid,
                                "data_source": "preliminary_shots",
                                "fixture_id": fixture_id,
                                "match_date_utc": match_data["match_date"],
                            })

            except Exception as e:
                logger.error(f"Error procesando fixture: {e}")
                continue

        # 4. Filtrar candidatos
        # 1x2: solo prob >= umbral. Las odds reales de bookmaker no siempre están
        #       disponibles en tiempo real, así que el EV calculado con defaults
        #       genéricos (2.0/3.3/3.5) es una ilusión matemática — no filtramos por él.
        # shots/corners: sí requieren EV > 0 porque usan odds más realistas (1.75/1.9).
        min_prob = max(threshold, 0.50)

        # Diagnóstico: muestra TODOS los candidatos antes del filtro
        logger.info(f"=== DISTRIBUCIÓN CANDIDATOS ({len(all_candidates)} total) — umbral={min_prob:.0%} ===")
        if not all_candidates:
            logger.warning("  Sin candidatos — verificar API, modelos y ligas activas")
        by_market = {}
        for c in all_candidates:
            by_market.setdefault(c["market_type"], []).append(c)
        for mtype, cands in by_market.items():
            cands_sorted = sorted(cands, key=lambda x: x["probability"], reverse=True)
            logger.info(f"  [{mtype}] {len(cands)} candidatos:")
            for c in cands_sorted[:5]:
                pass_prob = c["probability"] >= min_prob
                if mtype == "1x2":
                    passes = pass_prob
                    reason = "PASA" if passes else f"FILTRADO(prob={'OK' if pass_prob else 'BAJA'})"
                else:
                    pass_ev = c["expected_value"] > 0
                    passes = pass_prob and pass_ev
                    reason = "PASA" if passes else f"FILTRADO(prob={'OK' if pass_prob else 'BAJA'},ev={'OK' if pass_ev else 'NEG'})"
                logger.info(
                    f"    {reason} | prob={c['probability']:.1%} ev={c['expected_value']:+.3f} | "
                    f"{c['home_team']} vs {c['away_team']} | {c['prediction'][:50]}"
                )

        filtered = [
            c for c in all_candidates
            if c["probability"] >= min_prob
            and (c["market_type"] == "1x2" or c["expected_value"] > 0)
        ]

        # 5. Ordenar por probabilidad descendente
        selected = sorted(filtered, key=lambda x: x["probability"], reverse=True)

        logger.info(
            f"Candidatos: {len(all_candidates)} | "
            f"Filtrados (1x2: prob>={min_prob:.0%} | shots/corners: +EV>0): {len(filtered)} | "
            f"Seleccionados: {len(selected)}"
        )

        # 6. Guardar en BD
        model_version = get_model_version("1x2")
        api_source = "api_real" if api.api_healthy else "api_degraded"
        for pred in selected:
            # Preserve 'preliminary_shots' source; fall back to api health status
            effective_source = pred.get("data_source") or api_source
            insert_prediction({
                "match_id": pred["match_id"],
                "prediction_date": today,
                "market_type": pred["market_type"],
                "prediction": pred["prediction"],
                "probability": pred["probability"],
                "odds": pred["odds"],
                "expected_value": pred["expected_value"],
                "model_version": model_version,
                "data_source": effective_source,
                "player_id": pred.get("player_id"),
            })

        # 7. Fase 2: programar verificaciones de lineup para shots preliminares
        _schedule_lineup_checks(selected)

        # 8. Enviar Telegram
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
