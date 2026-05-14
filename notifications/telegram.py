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
            logger.warning("Error Markdown Telegram, reintentando sin formato")
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


def _split_and_send(header_lines, pred_lines, footer_lines, max_chars=4000):
    """Send predictions in chunks if they exceed Telegram's 4096-char limit."""
    chunks = []
    current = list(header_lines)
    for line in pred_lines:
        if len("\n".join(current + [line])) > max_chars:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    current.extend(footer_lines)
    chunks.append("\n".join(current))
    ok = True
    for msg in chunks:
        ok = send_telegram(msg) and ok
    return ok


def send_prediction_message(predictions: list, model_version: str, threshold: float,
                            cumulative_stats: dict, streak: int, worst_streak: int):
    from utils.helpers import today_colombia
    fecha = today_colombia().strftime("%d/%m/%Y")

    market_emoji = {"1x2": "⚽", "corners": "\U0001f532", "corners_stats": "🔢", "player_shots": "\U0001f3af"}

    header = [
        f"\U0001f3df *Predicciones {fecha}*",
        f"\U0001f4ca Modelo v{model_version} | Umbral: {int(threshold * 100)}% | "
        f"Total: {len(predictions)}",
        "",
    ]

    pred_lines = []
    for i, pred in enumerate(predictions, 1):
        emoji = market_emoji.get(pred["market_type"], "⚽")
        league = pred.get("league_name", "")
        league_str = f" _{league}_" if league else ""
        pred_lines.append(
            f"{i}. {emoji}{league_str} *{pred['home_team']}* vs *{pred['away_team']}*"
        )
        pred_lines.append(
            f"   ➡️ {pred['prediction']} | "
            f"Prob: *{int(pred['probability'] * 100)}%* | "
            f"Cuota: {pred['odds']:.2f} | "
            f"EV: +{pred['expected_value']:.2f}"
        )

    cum_total = cumulative_stats.get("total", 0)
    cum_correct = cumulative_stats.get("correct", 0)
    cum_pct = (cum_correct / cum_total * 100) if cum_total > 0 else 0

    footer = [
        "",
        f"\U0001f4c8 *Acumulado*: {cum_correct}/{cum_total} ({cum_pct:.1f}%)",
        f"\U0001f525 Racha: {abs(streak)} {'aciertos' if streak >= 0 else 'fallos'} | "
        f"Peor racha: {worst_streak} fallos",
    ]

    return _split_and_send(header, pred_lines, footer)


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
            emoji = "✅"
            suffix = "¡ACERTO!"
        else:
            emoji = "❌"
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
