"""
Evaluación incremental de predicciones.

Corre cada 30 min (2PM–12:30AM Colombia). Para cada predicción pendiente:
  1. Agrupa predicciones por fixture_id (1 llamada API por partido, no por predicción)
  2. Si no terminó o datos incompletos → skip (queda pending)
  3. Si terminó con datos completos → evalúa, envía Telegram individual
  4. Cuando TODAS las predicciones de un día se resuelven → consolida + retrain
"""
from collections import defaultdict
from datetime import date, timedelta
from data.api_football import (
    ApiFootballClient, parse_fixture, parse_fixture_statistics,
    parse_player_fixture_stats,
)
from db.models import (
    get_recent_pending_predictions, get_predictions_by_date,
    update_prediction_result, upsert_match,
    upsert_daily_performance, get_cumulative_stats,
    get_accuracy_by_type, fetch_one, void_stale_shot_predictions,
)
from models.expected_value import roi_from_predictions
from models.evaluator import get_current_threshold, get_model_version
from notifications.telegram import send_telegram, send_results_message
from config.settings import SIMULATED_STAKE
from utils.helpers import today_colombia
from utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation_check():
    """Chequeo periódico: evalúa predicciones pendientes que ya tengan datos.

    Agrupa predicciones por fixture_id para hacer UNA sola llamada
    get_fixture_by_id + get_fixture_statistics por partido (no por predicción).
    """
    logger.info("=== CHEQUEO DE RESULTADOS ===")

    try:
        api = ApiFootballClient()
        pending = get_recent_pending_predictions(lookback_days=7)

        if not pending:
            logger.info("No hay predicciones pendientes")
            for delta in range(7):
                _check_and_consolidate(today_colombia() - timedelta(days=delta))
            return

        logger.info(f"{len(pending)} predicciones pendientes")
        newly_evaluated = []

        api_status = api.check_status()
        if not api_status["ok"]:
            if api_status.get("quota_exhausted"):
                logger.warning("Cuota diaria de API-Football agotada, saltando evaluacion")
                # Telegram ya fue enviado dentro de check_status()
            else:
                logger.error("API-Football no responde, saltando evaluacion")
                send_telegram(
                    "ALERTA: API-Football caida durante evaluacion. "
                    "No se evaluaron predicciones en este ciclo.",
                    parse_mode=None,
                )
            return

        # Agrupar por fixture_id: 1 llamada API por partido, no por predicción
        by_fixture: dict = defaultdict(list)
        for pred in pending:
            fixture_id = pred.get("api_fixture_id")
            if not fixture_id:
                logger.warning(f"Predicción {pred['id']} sin api_fixture_id, skip")
                continue
            by_fixture[fixture_id].append(pred)

        logger.info(
            f"Agrupadas {len(pending)} predicciones en {len(by_fixture)} fixtures únicos"
        )

        for fixture_id, preds in by_fixture.items():
            try:
                # UNA sola llamada get_fixture_by_id por fixture
                fixture_data = api.get_fixture_by_id(fixture_id)
                if not fixture_data:
                    logger.warning(f"Fixture {fixture_id} no encontrado en API, skip")
                    continue

                match_data = parse_fixture(fixture_data)

                if match_data["status"] != "finished":
                    logger.info(
                        f"  {preds[0]['home_team']} vs {preds[0]['away_team']}: "
                        f"status={match_data['status']}, esperando..."
                    )
                    continue

                if match_data["home_score"] is None or match_data["away_score"] is None:
                    logger.warning(
                        f"  {preds[0]['home_team']} vs {preds[0]['away_team']}: "
                        f"finished pero sin scores, esperando actualización API"
                    )
                    continue

                # UNA sola llamada get_fixture_statistics por fixture
                stats_data = api.get_fixture_statistics(fixture_id)
                if stats_data:
                    parse_fixture_statistics(stats_data, match_data)

                upsert_match(match_data)

                for pred in preds:
                    try:
                        market = pred.get("market_type")

                        if market in ("corners", "corners_stats"):
                            if match_data.get("home_corners") is None or match_data.get("away_corners") is None:
                                logger.warning(
                                    f"  {pred['home_team']} vs {pred['away_team']}: "
                                    f"sin stats de corners, skip"
                                )
                                continue

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

                        pred["home_score"] = match_data["home_score"]
                        pred["away_score"] = match_data["away_score"]
                        pred["home_corners"] = match_data.get("home_corners")
                        pred["away_corners"] = match_data.get("away_corners")
                        pred["home_shots_on_target"] = match_data.get("home_shots_on_target")
                        pred["away_shots_on_target"] = match_data.get("away_shots_on_target")
                        pred["status"] = "finished"

                        result, actual = _evaluate_prediction(pred)
                        if result is None:
                            logger.warning(f"  Predicción {pred['id']}: no se pudo evaluar ({actual})")
                            continue

                        update_prediction_result(pred["id"], result, actual)
                        newly_evaluated.append({**pred, "result": result, "actual_outcome": actual})

                        emoji = "✅" if result == "win" else "❌"
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

            except Exception as e:
                logger.error(f"Error procesando fixture {fixture_id}: {e}")
                continue

        if newly_evaluated:
            logger.info(f"Evaluadas {len(newly_evaluated)} predicciones en este ciclo")

        # Consolidar todos los días con pendientes en la ventana de 7 días
        for delta in range(7):
            _check_and_consolidate(today_colombia() - timedelta(days=delta))

        logger.info("=== CHEQUEO DE RESULTADOS COMPLETADO ===")

    except Exception as e:
        logger.error(f"Error en chequeo de resultados: {e}")
        send_telegram(f"⚠️ Error en evaluación: {str(e)[:200]}")


def _check_and_consolidate(check_date: date):
    """Si todas las predicciones de check_date están resueltas (o void), envía
    consolidado y dispara re-entrenamiento.

    FIX A: predicciones de player_shots sin stats tras 26h se marcan 'void'
    automáticamente para no bloquear la consolidación. Void no cuenta para
    accuracy ni ROI.
    """
    # Step 1: auto-void shots sin resolver tras 26h
    voided = void_stale_shot_predictions(check_date)
    if voided > 0:
        logger.info(f"  {voided} predicciones de shots marcadas como void ({check_date})")

    all_preds = get_predictions_by_date(check_date)
    if not all_preds:
        return

    # Step 2: bloquear solo si quedan predicciones genuinamente sin resolver
    # (void = result set, no bloquea)
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

    # Solo win/loss cuentan para accuracy y ROI — void se excluye
    evaluated = [p for p in all_preds if p.get("result") in ("win", "loss")]
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
        send_telegram(f"⚠️ Error en re-entrenamiento: {str(e)[:200]}")


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

    elif market == "corners_stats":
        home_corners = pred.get("home_corners")
        away_corners = pred.get("away_corners")
        if home_corners is None or away_corners is None:
            return None, "no_corners_data"
        total_corners = home_corners + away_corners
        actual = f"{total_corners} corners"
        try:
            line = float(prediction.split("over ")[1].split(" ")[0])
            won = total_corners > line
        except (ValueError, IndexError):
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

        # Extraer línea del texto de predicción (e.g., "+0.5 disparos")
        line = 0.5
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

    # Extraer nombre del jugador del texto: "Vinícius Júnior +0.5 disparos a puerta"
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

    # Return None instead of 0/0 — caller will keep prediction pending
    # rather than incorrectly marking it as a loss (0 SOT, 0 min = "didn't play")
    logger.warning(f"Jugador '{player_name}' no encontrado en fixture {fixture_id} — dejando pendiente")
    return None


def _get_1x2_result(home_score, away_score, home_team, away_team):
    if home_score > away_score:
        return f"{home_team} gano {home_score}-{away_score}"
    elif home_score < away_score:
        return f"{away_team} gano {away_score}-{home_score}"
    else:
        return f"Empate {home_score}-{away_score}"
