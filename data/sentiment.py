"""
Analisis de sentimiento basico de noticias deportivas.
Usa keywords para detectar: lesiones, suspensiones, cambios de entrenador, etc.
Score de -1 (muy negativo) a 1 (muy positivo).
"""
from db.models import insert_sentiment
from utils.logger import get_logger

logger = get_logger(__name__)

# Keywords con pesos
NEGATIVE_KEYWORDS = {
    # Lesiones
    "injury": -0.8, "injured": -0.8, "lesion": -0.8, "lesionado": -0.8,
    "sidelined": -0.7, "out for": -0.7, "ruled out": -0.9,
    "hamstring": -0.6, "knee": -0.6, "ankle": -0.5, "fracture": -0.9,
    # Suspensiones
    "suspended": -0.7, "suspension": -0.7, "red card": -0.6,
    "ban": -0.6, "sancion": -0.7, "expulsado": -0.6,
    # Malos resultados
    "defeat": -0.4, "loss": -0.3, "derrota": -0.4, "crisis": -0.6,
    "sacked": -0.7, "fired": -0.7, "destituido": -0.7,
    # Conflictos
    "controversy": -0.3, "dispute": -0.3, "conflict": -0.3,
}

POSITIVE_KEYWORDS = {
    "win": 0.3, "victory": 0.4, "victoria": 0.4,
    "return": 0.5, "recovered": 0.6, "recuperado": 0.6, "fit": 0.5,
    "signing": 0.4, "fichaje": 0.4, "refuerzo": 0.4,
    "record": 0.3, "form": 0.3, "streak": 0.3,
    "unbeaten": 0.5, "invicto": 0.5,
    "clean sheet": 0.3, "goal": 0.2, "scored": 0.2,
}

KEY_ABSENCE_KEYWORDS = [
    "injury", "injured", "lesion", "lesionado", "sidelined",
    "ruled out", "out for", "suspended", "suspension", "ban",
    "sancion", "missing", "ausente", "baja",
]


def analyze_headline(headline: str, summary: str = "") -> dict:
    text = f"{headline} {summary}".lower()

    score = 0.0
    key_info_parts = []
    key_player_absent = False
    total_weights = 0

    for keyword, weight in NEGATIVE_KEYWORDS.items():
        if keyword in text:
            score += weight
            total_weights += 1
            key_info_parts.append(keyword)

    for keyword, weight in POSITIVE_KEYWORDS.items():
        if keyword in text:
            score += weight
            total_weights += 1

    if total_weights > 0:
        score = max(-1.0, min(1.0, score / total_weights))

    for keyword in KEY_ABSENCE_KEYWORDS:
        if keyword in text:
            key_player_absent = True
            break

    return {
        "sentiment_score": round(score, 2),
        "key_info": ", ".join(key_info_parts) if key_info_parts else None,
        "key_player_absent": key_player_absent,
    }


def analyze_team_news(team_id: int, team_name: str, news_items: list) -> dict:
    """
    Analiza noticias de un equipo y guarda en BD.
    Retorna score promedio y si hay jugador clave ausente.
    """
    if not news_items:
        return {"sentiment_score": 0.0, "key_player_absent": False}

    scores = []
    any_absent = False

    for item in news_items:
        analysis = analyze_headline(item.get("headline", ""), item.get("summary", ""))
        scores.append(analysis["sentiment_score"])
        if analysis["key_player_absent"]:
            any_absent = True

        try:
            insert_sentiment({
                "team_id": team_id,
                "team_name": team_name,
                "headline": item.get("headline", "")[:500],
                "source": item.get("source", ""),
                "sentiment_score": analysis["sentiment_score"],
                "key_info": analysis["key_info"],
                "published_at": item.get("published_at"),
            })
        except Exception as e:
            logger.error(f"Error guardando sentimiento: {e}")

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "sentiment_score": round(avg_score, 2),
        "key_player_absent": any_absent,
    }
