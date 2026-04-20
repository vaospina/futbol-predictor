"""
Recolector de cuotas/odds via API-Football.
"""
from data.api_football import ApiFootballClient, parse_odds
from db.models import insert_odds, get_odds_for_match
from utils.logger import get_logger

logger = get_logger(__name__)


def collect_odds_for_fixture(api: ApiFootballClient, fixture_id: int, match_id: int) -> dict:
    """Fetch and store odds for a fixture. Returns parsed odds dict."""
    odds_data = api.get_fixture_odds(fixture_id)
    if not odds_data:
        logger.debug(f"No odds para fixture {fixture_id}")
        return {}

    parsed = parse_odds(odds_data)
    if not parsed:
        return {}

    bookmaker = parsed.get("bookmaker", "Unknown")

    # Save key markets to odds_history
    markets_to_save = [
        ("Match Winner", "Home", parsed.get("odds_home_win")),
        ("Match Winner", "Draw", parsed.get("odds_draw")),
        ("Match Winner", "Away", parsed.get("odds_away_win")),
        ("Goals Over/Under", "Over 2.5", parsed.get("odds_over25")),
        ("Goals Over/Under", "Under 2.5", parsed.get("odds_under25")),
        ("Goals Over/Under", "Over 1.5", parsed.get("odds_over15")),
        ("Goals Over/Under", "Under 1.5", parsed.get("odds_under15")),
        ("Goals Over/Under", "Over 3.5", parsed.get("odds_over35")),
        ("Goals Over/Under", "Under 3.5", parsed.get("odds_under35")),
        ("Both Teams Score", "Yes", parsed.get("odds_btts_yes")),
        ("Both Teams Score", "No", parsed.get("odds_btts_no")),
    ]

    saved = 0
    for market, selection, odds_value in markets_to_save:
        if odds_value and odds_value > 0:
            try:
                insert_odds({
                    "match_id": match_id,
                    "bookmaker": bookmaker,
                    "market": market,
                    "selection": selection,
                    "odds_value": odds_value,
                })
                saved += 1
            except Exception as e:
                logger.debug(f"Error guardando odds: {e}")

    logger.info(f"Odds fixture {fixture_id}: {saved} entries ({bookmaker})")
    return parsed


def get_match_odds_summary(match_id: int) -> dict:
    """Returns odds summary for a match from DB."""
    odds = get_odds_for_match(match_id)

    summary = {}
    for o in odds:
        market = o.get("market", "")
        selection = o.get("selection", "")
        val = o.get("odds_value", 0)

        if market == "Match Winner":
            if selection == "Home":
                summary["odds_home_win"] = val
            elif selection == "Draw":
                summary["odds_draw"] = val
            elif selection == "Away":
                summary["odds_away_win"] = val
        elif market == "Goals Over/Under":
            if selection == "Over 1.5":
                summary["odds_over15"] = val
            elif selection == "Under 1.5":
                summary["odds_under15"] = val
            elif selection == "Over 2.5":
                summary["odds_over25"] = val
            elif selection == "Under 2.5":
                summary["odds_under25"] = val
            elif selection == "Over 3.5":
                summary["odds_over35"] = val
            elif selection == "Under 3.5":
                summary["odds_under35"] = val
        elif market == "Both Teams Score":
            if selection == "Yes":
                summary["odds_btts_yes"] = val
            elif selection == "No":
                summary["odds_btts_no"] = val
        elif "corner" in market.lower() and "over" in selection.lower():
            summary["odds_over_corners"] = val

    return summary
