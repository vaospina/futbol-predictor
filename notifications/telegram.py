import requests
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import get_logger

logger = get_logger(__name__)


def send_telegram(message: str, parse_mode: str = "Markdown"):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Mensaje Telegram enviado exitosamente")
            return True
        elif response.status_code == 400 and parse_mode:
            logger.warning(f"Error Markdown Telegram, reintentando sin formato")
            payload.pop("parse_mode", None)
            retry = requests.post(url, json=payload, timeout=10)
            if retry.status_code == 200:
                logger.info("Mensaje Telegram enviado (sin formato)")
                return True
            logger.error(f"Error Telegram retry ({retry.status_code}): {retry.text}")
            return False
        else:
            logger.error(f"Error Telegram ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error enviando Telegram: {e}")
        return False


def send_prediction_message(predictions: list, model_version: str, threshold: float,
                            cumulative_stats: dict, streak: int, worst_streak: int):
    from utils.helpers import today_colombia
    fecha = today_colombia().strftime("%d/%m/%Y")

    lines = [
        f"\U0001f3df *Predicciones {fecha}*",
        f"\U0001f4ca Modelo v{model_version} | Umbral: {int(threshold * 100)}%",
        "",
    ]

    for i, pred in enumerate(predictions, 1):
        emoji = {
            "1x2": "\u26bd",
            "corners": "\U0001f532",
            "player_shots": "\U0001f3af",
        }.get(pred["market_type"], "\u26bd")

        lines.append(
            f"{i}. {emoji} {pred['home_team']} vs {pred['away_team']} — *{pred['prediction']}*"
        )
        lines.append(
            f"   Prob: {int(pred['probability'] * 100)}% | "
            f"Cuota: {pred['odds']:.2f} | "
            f"EV: +{pred['expected_value']:.2f}"
        )

    cum_total = cumulative_stats.get("total", 0)
    cum_correct = cumulative_stats.get("correct", 0)
    cum_pct = (cum_correct / cum_total * 100) if cum_total > 0 else 0

    lines.extend([
        "",
        f"\U0001f4c8 *Rendimiento acumulado*: {cum_correct}/{cum_total} ({cum_pct:.1f}%)",
        f"\U0001f525 Racha actual: {abs(streak)} {'aciertos' if streak >= 0 else 'fallos'}",
        f"\U0001f4c9 Peor racha: {worst_streak} fallos seguidos",
    ])

    return send_telegram("\n".join(lines))


def send_results_message(results: list, day_stats: dict, cumulative_stats: dict,
                         accuracy_by_type: dict, roi: float, model_version: str):
    from utils.helpers import today_colombia
    fecha = today_colombia().strftime("%d/%m/%Y")

    lines = [
        f"\U0001f319 *Resultados {fecha}*",
        "",
    ]

    for i, r in enumerate(results, 1):
        if r["result"] == "win":
            emoji = "\u2705"
            suffix = "\u00a1ACERTO!"
        else:
            emoji = "\u274c"
            suffix = "FALLO"

        lines.append(
            f"{i}. {emoji} {r['home_team']} {r.get('home_score', '?')}-"
            f"{r.get('away_score', '?')} {r['away_team']} — "
            f"*{r['prediction']}* {suffix}"
        )

    day_total = day_stats.get("total", 0)
    day_correct = day_stats.get("correct", 0)
    day_pct = (day_correct / day_total * 100) if day_total > 0 else 0

    cum_total = cumulative_stats.get("total", 0)
    cum_correct = cumulative_stats.get("correct", 0)
    cum_pct = (cum_correct / cum_total * 100) if cum_total > 0 else 0

    lines.extend([
        "",
        f"\U0001f4ca *Hoy*: {day_correct}/{day_total} ({day_pct:.1f}%)",
        f"\U0001f4c8 *Acumulado*: {cum_correct}/{cum_total} ({cum_pct:.1f}%)",
    ])

    if accuracy_by_type:
        type_lines = []
        for t in accuracy_by_type:
            type_lines.append(f"{t['market_type']}: {t['accuracy']}%")
        lines.append(f"\U0001f4c9 *Accuracy por tipo*:")
        lines.append(f"   {' | '.join(type_lines)}")

    lines.append(f"\U0001f4b0 *ROI simulado*: {roi:+.1f}% (stake fijo $10K COP)")
    lines.append(f"\U0001f504 *Modelo*: v{model_version}")

    return send_telegram("\n".join(lines))
