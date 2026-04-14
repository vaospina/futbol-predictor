"""
Flujo de la noche (10:00 PM hora Colombia / 03:00 UTC).
Evalua resultados del dia y actualiza rendimiento.
"""
from datetime import date
from data.api_football import ApiFootballClient, parse_fixture, parse_fixture_statistics
from db.models import (
    get_pending_predictions, update_prediction_result,
    upsert_match, upsert_daily_performance,
    get_cumulative_stats, get_accuracy_by_type, get_config,
    get_predictions_by_date,
)
from models.expected_value import roi_from_predictions
from models.evaluator import get_current_threshold, get_model_version
from notifications.telegram import send_results_message, send_telegram
from config.settings import SIMULATED_STAKE
from utils.helpers import today_colombia
from utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_results():
    """Flujo completo de evaluacion de resultados."""
    logger.info("=== INICIANDO FLUJO DE RESULTADOS ===")

    try:
        api = ApiFootballClient()
        today = today_colombia()

        # 1. Obtener predicciones pendientes del dia
        pending = get_pending_predictions(today)
        if not pending:
            logger.info("No hay predicciones pendientes para hoy")
            return

        logger.info(f"Evaluando {len(pending)} predicciones")

        results_list = []

        # 2. Para cada prediccion, verificar resultado
        for pred in pending:
            try:
                fixture_id = pred.get("api_fixture_id")

                # Obtener resultado actualizado de la API
                if fixture_id:
                    stats_data = api.get_fixture_statistics(fixture_id)

                    # Actualizar match en BD con stats finales
                    match_update = {
                        "api_fixture_id": fixture_id,
                        "league_id": pred.get("league_id", 0),
                        "league_name": pred.get("league_name", ""),
                        "season": pred.get("season", 2025),
                        "match_date": pred.get("match_date", today),
                        "home_team": pred["home_team"],
                        "away_team": pred["away_team"],
                        "home_team_id": pred.get("home_team_id"),
                        "away_team_id": pred.get("away_team_id"),
                        "home_score": pred.get("home_score"),
                        "away_score": pred.get("away_score"),
                        "status": pred.get("status", "finished"),
                        "home_corners": pred.get("home_corners"),
                        "away_corners": pred.get("away_corners"),
                        "home_shots_on_target": pred.get("home_shots_on_target"),
                        "away_shots_on_target": pred.get("away_shots_on_target"),
                        "home_possession": None,
                        "away_possession": None,
                        "venue": None,
                    }

                    if stats_data:
                        match_update = parse_fixture_statistics(stats_data, match_update)

                    upsert_match(match_update)

                # Evaluar prediccion
                result, actual = _evaluate_prediction(pred)

                if result:
                    update_prediction_result(pred["id"], result, actual)

                results_list.append({
                    **pred,
                    "result": result,
                    "actual_outcome": actual,
                })

                logger.info(
                    f"  {pred['home_team']} vs {pred['away_team']} | "
                    f"{pred['prediction']} -> {result} ({actual})"
                )

            except Exception as e:
                logger.error(f"Error evaluando prediccion {pred.get('id')}: {e}")
                continue

        # 3. Calcular metricas del dia
        day_correct = sum(1 for r in results_list if r.get("result") == "win")
        day_total = len(results_list)
        day_accuracy = day_correct / day_total if day_total > 0 else 0

        # 4. Actualizar rendimiento acumulado
        cum_stats = get_cumulative_stats()
        cum_total = (cum_stats.get("total", 0) if cum_stats else 0) + day_total
        cum_correct = (cum_stats.get("correct", 0) if cum_stats else 0) + day_correct
        cum_accuracy = cum_correct / cum_total if cum_total > 0 else 0

        # ROI simulado
        all_preds_today = get_predictions_by_date(today)
        roi = roi_from_predictions(all_preds_today, SIMULATED_STAKE)

        model_version = get_model_version("1x2")
        threshold = get_current_threshold()

        upsert_daily_performance({
            "date": today,
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

        # 5. Enviar Telegram con resultados
        accuracy_by_type = get_accuracy_by_type()

        send_results_message(
            results=results_list,
            day_stats={"total": day_total, "correct": day_correct},
            cumulative_stats={"total": cum_total, "correct": cum_correct},
            accuracy_by_type=accuracy_by_type,
            roi=roi,
            model_version=model_version,
        )

        logger.info(
            f"=== RESULTADOS: {day_correct}/{day_total} ({day_accuracy:.0%}) | "
            f"Acumulado: {cum_correct}/{cum_total} ({cum_accuracy:.0%}) ==="
        )

    except Exception as e:
        logger.error(f"Error en flujo de resultados: {e}")
        send_telegram(f"\u26a0\ufe0f Error en resultados: {str(e)[:200]}")


def _evaluate_prediction(pred: dict) -> tuple:
    """
    Evalua una prediccion vs resultado real.
    Retorna (result, actual_outcome) donde result es 'win' o 'loss'.
    """
    market = pred.get("market_type")
    prediction = pred.get("prediction", "")
    status = pred.get("status", "")

    if status != "finished":
        return None, "match_not_finished"

    home_score = pred.get("home_score")
    away_score = pred.get("away_score")

    if home_score is None or away_score is None:
        return None, "no_score"

    if market == "1x2":
        actual = _get_1x2_result(home_score, away_score, pred["home_team"], pred["away_team"])
        prediction_lower = prediction.lower()
        if "gana" in prediction_lower:
            # Extraer equipo de la prediccion
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
            # Over: +9.5 corners
            try:
                line = float(prediction.split("+")[1].split(" ")[0])
                won = total_corners > line
            except (ValueError, IndexError):
                won = False
        elif "-" in prediction:
            # Under: -9.5 corners
            try:
                line = float(prediction.split("-")[1].split(" ")[0])
                won = total_corners < line
            except (ValueError, IndexError):
                won = False
        else:
            won = False

        return ("win" if won else "loss"), actual

    elif market == "player_shots":
        # Para shots, necesitamos datos del jugador del partido
        # Por ahora evaluacion basica
        home_sot = pred.get("home_shots_on_target") or 0
        away_sot = pred.get("away_shots_on_target") or 0
        actual = f"SOT: {home_sot}-{away_sot}"

        # Evaluacion simplificada: si dice +1.5 disparos
        if "+1.5" in prediction:
            # Asumimos que si el equipo tiene >= 2 SOT, el top shooter probablemente tuvo >= 2
            won = max(home_sot, away_sot) >= 2
        else:
            won = False

        return ("win" if won else "loss"), actual

    return None, "unknown_market"


def _get_1x2_result(home_score, away_score, home_team, away_team):
    if home_score > away_score:
        return f"{home_team} gano {home_score}-{away_score}"
    elif home_score < away_score:
        return f"{away_team} gano {away_score}-{home_score}"
    else:
        return f"Empate {home_score}-{away_score}"
