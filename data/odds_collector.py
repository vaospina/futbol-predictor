"""
Recolector de cuotas/odds via API-Football.
"""
from data.api_football import ApiFootballClient
from db.models import insert_odds, get_match_by_fixture_id
from utils.logger import get_logger

logger = get_logger(__name__)

api = ApiFootballClient()


def collect_odds_for_fixture(fixture_id: int, match_id: int):
    odds_data = api.get_odds(fixture_id)
    if not odds_data:
        logger.debug(f"No odds encontradas para fixture {fixture_id}")
        return []

    saved = []
    for response in odds_data:
        for bookmaker in response.get("bookmakers", []):
            bk_name = bookmaker.get("name", "Unknown")
            for bet in bookmaker.get("bets", []):
                market = bet.get("name", "")
                for value in bet.get("values", []):
                    odds_entry = {
                        "match_id": match_id,
                        "bookmaker": bk_name,
                        "market": market,
                        "selection": value.get("value", ""),
                        "odds_value": float(value.get("odd", 0)),
                    }
                    try:
                        insert_odds(odds_entry)
                        saved.append(odds_entry)
                    except Exception as e:
                        logger.error(f"Error guardando odds: {e}")

    logger.info(f"Guardadas {len(saved)} odds para fixture {fixture_id}")
    return saved


def get_match_odds_summary(match_id: int) -> dict:
    """Retorna un resumen de odds 1X2 y corners para un partido."""
    from db.models import get_odds_for_match
    odds = get_odds_for_match(match_id)

    summary = {
        "odds_home_win": None,
        "odds_draw": None,
        "odds_away_win": None,
        "odds_over_corners": None,
    }

    for o in odds:
        market = o["market"].lower() if o.get("market") else ""
        selection = o["selection"].lower() if o.get("selection") else ""

        if "winner" in market or "1x2" in market:
            if selection in ("home", "1"):
                summary["odds_home_win"] = o["odds_value"]
            elif selection in ("draw", "x"):
                summary["odds_draw"] = o["odds_value"]
            elif selection in ("away", "2"):
                summary["odds_away_win"] = o["odds_value"]

        if "corner" in market and "over" in selection:
            summary["odds_over_corners"] = o["odds_value"]

    return summary
