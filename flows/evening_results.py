"""
Evaluación incremental de predicciones.

Corre cada 30 min (2PM–12:30AM Colombia). Para cada predicción pendiente:
  1. Re-fetch del fixture desde API-Football (score + status)
  2. Si no terminó o datos incompletos → skip (queda pending)
  3. Si terminó con datos completos → evalúa, envía Telegram individual
  4. Cuando TODAS las predicciones de un día se resuelven → consolida + retrain
"""
from datetime import date, timedelta
from data.api_football import (
    ApiFootballClient, parse_fixture, parse_fixture_statistics,
    parse_player_fixture_stats,
)
from db.models import (
    get_recent_pending_predictions, get_predictions_by_date,
    update_prediction_result, upsert_match,
    upsert_daily_performance, get_cumulative_stats,
    get_accuracy_by_type, fetch_one,
)
from models.expected_value import roi_from_predictions
from models.evaluator import get_current_threshold, get_model_version
from notifications.telegram import send_telegram, send_results_message
from config.settings import SIMULATED_STAKE
from utils.helpers import today_colombia
from utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation_check():
    """Chequeo periódico: evalúa predicciones pendientes que ya tengan datos."""
    logger.info("=== CHEQUEO DE RESULTADOS ===")

    try:
        api = ApiFootballClient()
        pending = get_recent_pending_predictions(lookback_days=2)

        if not pending:
            logger.info("No hay predicciones pendientes")
            return

        logger.info(f"{len(pending)} predicciones pendientes")
        newly_evaluated = []

        for pred in pending:
            try:
                fixture_id = pred.get("api_fixture_id")
                if not fixture_id:
                    logger.warning(f"Predicción {pred['id']} sin api_fixture_id, skip")
                    continue

                # Re-fetch fixture desde la API (score + status actualizados)
                fixture_data = api.get_fixture_by_id(fixture_id)
                if not fixture_data:
                    logger.warning(f"Fixture {fixture_id} no encontrado en API, skip")
                    continue

                match_data = parse_fixture(fixture_data)

                if match_data["status"] != "finished":
                    logger.info(
                        f"  {pred['home_team']} vs {pred['away_team']}: "
                        f"status={match_data['status']}, esperando..."
                    )
                    continue

                if match_data["home_score"] is None or match_data["away_score"] is None:
                    logger.warning(
                        f"  {pred['home_team']} vs {pred['away_team']}: "
                        f"finished pero sin scores, esperando actualización API"
                    )
                    continue

                # Fetch stats (corners, shots, posesión)
                stats_data = api.get_fixture_statistics(fixture_id)
                if stats_data:
                    parse_fixture_statistics(stats_data, match_data)

                # Verificar datos completos según mercado
                market = pred.get("market_type")
                if market == "corners":
                    if match_data.get("home_corners") is None or match_data.get("away_corners") is None:
                        logger.warning(
                            f"  {pred['home_team']} vs {pred['away_team']}: "
                            f"sin stats de corners, skip"
                        )
                        continue

                # Para player_shots: obtener stats INDIVIDUALES del jugador
                if market == "player_shots":
                    player_sot = _fetch_player_sot(api, fixture_id, pred["prediction"])
                    if player_sot is None:
                        logger.warning(
                            f"  {pred['home_team']} vs {pred['away_team']}: "
                            f"sin stats individuales del jugador, skip"
                        )
                        continue
                    pred["player_sot"] = player_sot["sot"]
                    pred["player_minutes"] = player_sot["minutes"]
                    pred["player_name_found"] = player_sot["name"]

                # Actualizar match en BD con datos frescos
                upsert_match(match_data)

                # Merge datos frescos al dict de predicción para evaluar
                pred["home_score"] = match_data["home_score"]
                pred["away_score"] = match_data["away_score"]
                pred["home_corners"] = match_data.get("home_corners")
                pred["away_corners"] = match_data.get("away_corners")
                pred["home_shots_on_target"] = match_data.get("home_shots_on_target")
                pred["away_shots_on_target"] = match_data.get("away_shots_on_target")
                pred["status"] = "finished"

                # Evaluar
                result, actual = _evaluate_prediction(pred)
                if result is None:
                    logger.warning(f"  Predicción {pred['id']}: no se pudo evaluar ({actual})")
                    continue

                update_prediction_result(pred["id"], result, actual)
                newly_evaluated.append({**pred, "result": result, "actual_outcome": actual})

                # Telegram individual
                emoji = "\u2705" if result == "win" else "\u274c"
                score_str = f"{match_data['home_score']}-{match_data['away_score']}"
                prob_pct = int(pred.get("probability", 0) * 100)
                send_telegram(
                    f"{emoji} {pred['home_team']} {score_str} {pred['away_team']} "
                    f"— *{pred['prediction']}* {'WIN' if result == 'win' else 'LOSS'} "
                    f"| Prob era {prob_pct}%"
                )

                logger.info(
                    f"  {pred['home_team']} vs {pred['away_team']} | "
                    f"{pred['prediction']} -> {result} ({actual})"
                )

            except Exception as e:
                logger.error(f"Error evaluando predicción {pred.get('id')}: {e}")
                continue

        if newly_evaluated:
            logger.info(f"Evaluadas {len(newly_evaluated)} predicciones en este ciclo")

        # Verificar si hay días con TODAS las predicciones resueltas
        _check_and_consolidate(today_colombia())
        _check_and_consolidate(today_colombia() - timedelta(days=1))

        logger.info("=== CHEQUEO DE RESULTADOS COMPLETADO ===")

    except Exception as e:
        logger.error(f"Error en chequeo de resultados: {e}")
        send_telegram(f"\u26a0\ufe0f Error en evaluación: {str(e)[:200]}")


def _check_and_consolidate(check_date: date):
    """Si TODAS las predicciones de check_date están resueltas, envía
    consolidado y dispara re-entrenamiento."""
    all_preds = get_predictions_by_date(check_date)
    if not all_preds:
        return

    pending = [p for p in all_preds if p.get("result") is None]
    if pending:
        return

    # Verificar que no se haya consolidado ya
    existing = fetch_one(
        "SELECT id FROM daily_performance WHERE date = :d",
        {"d": check_date}
    )
    if existing:
        return

    logger.info(f"=== CONSOLIDANDO RESULTADOS {check_date} ===")

    evaluated = [p for p in all_preds if p.get("result") is not None]
    day_correct = sum(1 for p in evaluated if p["result"] == "win")
    day_total = len(evaluated)
    day_accuracy = day_correct / day_total if day_total > 0 else 0

    cum_stats = get_cumulative_stats()
    cum_total = (cum_stats.get("total", 0) if cum_stats else 0) + day_total
    cum_correct = (cum_stats.get("correct", 0) if cum_stats else 0) + day_correct
    cum_accuracy = cum_correct / cum_total if cum_total > 0 else 0

    roi = roi_from_predictions(evaluated, SIMULATED_STAKE)
    model_version = get_model_version("1x2")
    threshold = get_current_threshold()

    upsert_daily_performance({
        "date": check_date,
        "total_predictions": day_total,
        "correct_predictions": day_correct,
        "accuracy": day_accuracy,
        "cumulative_total": cum_total,
        "cumulative_correct": cum_correct,
        "cumulative_accuracy": cum_accuracy,
        "roi_simulated": roi,
        "model_version": model_version,
        "threshold_used": threshold,
    })

    accuracy_by_type = get_accuracy_by_type()
    send_results_message(
        results=evaluated,
        day_stats={"total": day_total, "correct": day_correct},
        cumulative_stats={"total": cum_total, "correct": cum_correct},
        accuracy_by_type=accuracy_by_type,
        roi=roi,
        model_version=model_version,
    )

    logger.info(
        f"Consolidado {check_date}: {day_correct}/{day_total} ({day_accuracy:.0%}) | "
        f"Acumulado: {cum_correct}/{cum_total} ({cum_accuracy:.0%})"
    )

    # Disparar re-entrenamiento diario
    try:
        from flows.daily_retrain import run_daily_retrain
        run_daily_retrain()
    except Exception as e:
        logger.error(f"Error en re-entrenamiento post-consolidado: {e}")
        send_telegram(f"\u26a0\ufe0f Error en re-entrenamiento: {str(e)[:200]}")


def _evaluate_prediction(pred: dict) -> tuple:
    """Evalúa una predicción vs resultado real.
    Retorna (result, actual_outcome). result='win'|'loss' o None si no evaluable."""
    market = pred.get("market_type")
    prediction = pred.get("prediction", "")

    home_score = pred.get("home_score")
    away_score = pred.get("away_score")

    if home_score is None or away_score is None:
        return None, "no_score"

    if market == "1x2":
        actual = _get_1x2_result(home_score, away_score, pred["home_team"], pred["away_team"])
        prediction_lower = prediction.lower()
        if "gana" in prediction_lower:
            if pred["home_team"].lower() in prediction_lower:
                won = home_score > away_score
            elif pred["away_team"].lower() in prediction_lower:
                won = away_score > home_score
            else:
                won = False
        elif "empate" in prediction_lower:
            won = home_score == away_score
        else:
            won = False
        return ("win" if won else "loss"), actual

    elif market == "corners":
        home_corners = pred.get("home_corners")
        away_corners = pred.get("away_corners")
        if home_corners is None or away_corners is None:
            return None, "no_corners_data"

        total_corners = home_corners + away_corners
        actual = f"{total_corners} corners"

        if "+" in prediction:
            try:
                line = float(prediction.split("+")[1].split(" ")[0])
                won = total_corners > line
            except (ValueError, IndexError):
                won = False
        elif "-" in prediction:
            try:
                line = float(prediction.split("-")[1].split(" ")[0])
                won = total_corners < line
            except (ValueError, IndexError):
                won = False
        else:
            won = False
        return ("win" if won else "loss"), actual

    elif market == "player_shots":
        player_sot = pred.get("player_sot")
        player_minutes = pred.get("player_minutes", 0)
        player_found = pred.get("player_name_found", "?")

        if player_sot is None:
            return None, "no_player_stats"

        if player_minutes is None or player_minutes == 0:
            logger.info(
                f"  Player {player_found}: mins=0, no jugó -> LOSS"
            )
            return "loss", f"{player_found}: no jugó (0 min)"

        # Extraer línea del texto de predicción (e.g., "+1.5 disparos")
        line = 1.5
        if "+0.5" in prediction:
            line = 0.5
        elif "+1.5" in prediction:
            line = 1.5
        elif "+2.5" in prediction:
            line = 2.5

        won = player_sot > line
        actual = f"{player_found}: SOT={player_sot}, line={line}"

        logger.info(
            f"  Player {player_found}: "
            f"shots_on_target={player_sot}, line={line}, "
            f"result={'WIN' if won else 'LOSS'}"
        )

        return ("win" if won else "loss"), actual

    return None, "unknown_market"


def _normalize(s: str) -> str:
    """Normaliza nombre quitando acentos y pasando a minúsculas."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _fetch_player_sot(api, fixture_id: int, prediction_text: str) -> dict | None:
    """Busca las stats individuales del jugador en el fixture vía API.
    Retorna {"name": ..., "sot": int, "minutes": int} o None."""

    # Extraer nombre del jugador del texto: "Vinícius Júnior +1.5 disparos a puerta"
    for sep in [" +0.5", " +1.5", " +2.5", " -0.5", " -1.5", " -2.5"]:
        if sep in prediction_text:
            player_name = prediction_text.split(sep)[0].strip()
            break
    else:
        logger.warning(f"No se pudo extraer nombre de jugador de: {prediction_text}")
        return None

    players_raw = api.get_fixture_player_stats(fixture_id)
    if not players_raw:
        return None

    players = parse_player_fixture_stats(players_raw)
    if not players:
        return None

    player_norm = _normalize(player_name)

    # Búsqueda exacta primero, luego parcial
    for p in players:
        if _normalize(p["player_name"]) == player_norm:
            return {"name": p["player_name"], "sot": p["shots_on_target"], "minutes": p["minutes_played"]}

    for p in players:
        api_norm = _normalize(p["player_name"])
        if player_norm in api_norm or api_norm in player_norm:
            return {"name": p["player_name"], "sot": p["shots_on_target"], "minutes": p["minutes_played"]}

    # Buscar por apellido (última palabra)
    surname = player_norm.split()[-1] if player_norm else ""
    if len(surname) > 2:
        for p in players:
            if surname in _normalize(p["player_name"]):
                return {"name": p["player_name"], "sot": p["shots_on_target"], "minutes": p["minutes_played"]}

    logger.warning(f"Jugador '{player_name}' no encontrado en fixture {fixture_id}")
    return {"name": player_name, "sot": 0, "minutes": 0}


def _get_1x2_result(home_score, away_score, home_team, away_team):
    if home_score > away_score:
        return f"{home_team} gano {home_score}-{away_score}"
    elif home_score < away_score:
        return f"{away_team} gano {away_score}-{home_score}"
    else:
        return f"Empate {home_score}-{away_score}"
