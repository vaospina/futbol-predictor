"""
Fase 2 de shots: verificación de lineups 60 min antes de cada partido.

Corre como job dinámico en APScheduler (programado por morning_predictions).
Confirma o cancela las predicciones de shots marcadas como 'preliminary_shots'.
"""
from data.api_football import ApiFootballClient, parse_lineups
from db.models import (
    get_preliminary_shots_by_match,
    update_prediction_source,
    delete_prediction,
)
from notifications.telegram import send_telegram
from utils.logger import get_logger

logger = get_logger(__name__)


def verify_shots_for_match(fixture_id: int, match_id: int, home_team: str, away_team: str):
    """Consulta lineups confirmados y actualiza predicciones de shots preliminares."""
    logger.info(f"=== VERIFICANDO LINEUPS: {home_team} vs {away_team} (fixture {fixture_id}) ===")

    preliminary = get_preliminary_shots_by_match(match_id)
    if not preliminary:
        logger.info(f"  No hay shots preliminares para match_id={match_id}")
        return

    api = ApiFootballClient()
    lineups_raw = api.get_fixture_lineups(fixture_id)

    if not lineups_raw:
        logger.info(f"  Lineups aún no disponibles para fixture {fixture_id} — se mantienen preliminares")
        return

    lineup_status = parse_lineups(lineups_raw)
    confirmed = []
    cancelled = []

    for pred in preliminary:
        player_id = pred.get("player_id")
        status = lineup_status.get(player_id) if player_id else None

        if status == "starter":
            update_prediction_source(pred["id"], "confirmed_shots")
            confirmed.append(pred)
            logger.info(f"  CONFIRMADO: {pred['prediction']}")
        else:
            reason = "en banco" if status == "bench" else "no convocado"
            delete_prediction(pred["id"])
            cancelled.append({**pred, "reason": reason})
            logger.info(f"  CANCELADO ({reason}): {pred['prediction']}")

    if not confirmed and not cancelled:
        return

    lines = [f"⚽ *Lineups — {home_team} vs {away_team}*", ""]
    for p in confirmed:
        lines.append(
            f"✅ *{p['prediction']}* — CONFIRMADO "
            f"(prob={int(p['probability'] * 100)}%)"
        )
    for p in cancelled:
        lines.append(f"\U0001f6ab {p['prediction']} — cancelado ({p['reason']})")

    send_telegram("\n".join(lines))
    logger.info(
        f"=== LINEUP CHECK COMPLETADO: {len(confirmed)} confirmados, "
        f"{len(cancelled)} cancelados ==="
    )
