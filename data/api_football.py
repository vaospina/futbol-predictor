"""
Cliente API-Football - Conexion directa (NO RapidAPI).
Header de autenticacion: x-apisports-key
"""
import requests
from datetime import date, datetime
from config.settings import API_FOOTBALL_KEY, API_FOOTBALL_BASE_URL, CURRENT_SEASON
from utils.logger import get_logger

logger = get_logger(__name__)

_FAILURE_COUNT = 0
_ALERT_SENT = False


class ApiFootballClient:
    def __init__(self):
        self.base_url = API_FOOTBALL_BASE_URL
        self.headers = {
            "x-apisports-key": API_FOOTBALL_KEY,
        }
        self.requests_today = 0
        self.api_healthy = True

    def _get(self, endpoint: str, params: dict = None) -> dict:
        global _FAILURE_COUNT, _ALERT_SENT
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            self.requests_today += 1
            if response.status_code == 200:
                data = response.json()
                if data.get("errors"):
                    logger.warning(f"API-Football errors: {data['errors']}")
                _FAILURE_COUNT = 0
                _ALERT_SENT = False
                self.api_healthy = True
                return data
            else:
                logger.error(f"API-Football error ({response.status_code}): {response.text}")
                self._track_failure(f"HTTP {response.status_code} en {endpoint}")
                return {"response": [], "_api_error": True}
        except Exception as e:
            logger.error(f"Error llamando API-Football {endpoint}: {e}")
            self._track_failure(f"{type(e).__name__}: {e}")
            return {"response": [], "_api_error": True}

    def _track_failure(self, reason: str):
        global _FAILURE_COUNT, _ALERT_SENT
        _FAILURE_COUNT += 1
        self.api_healthy = False
        if _FAILURE_COUNT >= 3 and not _ALERT_SENT:
            _ALERT_SENT = True
            try:
                from notifications.telegram import send_telegram
                send_telegram(
                    f"ALERTA: API-Football ha fallado {_FAILURE_COUNT} veces consecutivas.\n"
                    f"Ultimo error: {reason}\n"
                    f"Las predicciones pueden no ser confiables.",
                    parse_mode=None,
                )
            except Exception:
                pass

    def check_status(self) -> dict:
        data = self._get("status")
        response = data.get("response", {})
        return {
            "ok": not data.get("_api_error", False),
            "account": response.get("account", {}),
            "subscription": response.get("subscription", {}),
            "requests": response.get("requests", {}),
        }

    def get_fixtures_by_date(self, league_id: int, match_date: date, season: int = None):
        season = season or CURRENT_SEASON
        params = {
            "league": league_id,
            "season": season,
            "from": match_date.isoformat(),
            "to": match_date.isoformat(),
        }
        data = self._get("fixtures", params)
        return data.get("response", [])

    def get_fixtures_by_date_range(self, league_id: int, from_date: date, to_date: date, season: int = None):
        season = season or CURRENT_SEASON
        params = {
            "league": league_id,
            "season": season,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
        data = self._get("fixtures", params)
        return data.get("response", [])

    def get_fixture_statistics(self, fixture_id: int):
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixture_events(self, fixture_id: int):
        data = self._get("fixtures/events", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixtures_last_n(self, league_id: int, n: int = 10, season: int = None):
        season = season or CURRENT_SEASON
        params = {"league": league_id, "season": season, "last": n}
        data = self._get("fixtures", params)
        return data.get("response", [])

    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10):
        params = {"h2h": f"{team1_id}-{team2_id}", "last": last}
        data = self._get("fixtures/headtohead", params)
        return data.get("response", [])

    def get_standings(self, league_id: int, season: int = None):
        season = season or CURRENT_SEASON
        data = self._get("standings", {"league": league_id, "season": season})
        response = data.get("response", [])
        if response and response[0].get("league", {}).get("standings"):
            return response[0]["league"]["standings"][0]
        return []

    def get_fixture_odds(self, fixture_id: int):
        """Get odds for a fixture. Tries Bet365 first, falls back to any available."""
        data = self._get("odds", {"fixture": fixture_id})
        return data.get("response", [])

    def get_player_stats(self, player_id: int, season: int = None, league_id: int = None):
        season = season or CURRENT_SEASON
        params = {"id": player_id, "season": season}
        if league_id:
            params["league"] = league_id
        data = self._get("players", params)
        return data.get("response", [])

    def get_fixture_lineups(self, fixture_id: int):
        data = self._get("fixtures/lineups", {"fixture": fixture_id})
        return data.get("response", [])

    def get_predictions(self, fixture_id: int):
        data = self._get("predictions", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixture_player_stats(self, fixture_id: int):
        data = self._get("fixtures/players", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixture_by_id(self, fixture_id: int):
        data = self._get("fixtures", {"id": fixture_id})
        response = data.get("response", [])
        return response[0] if response else None

    def get_season_fixtures(self, league_id: int, season: int = None):
        season = season or CURRENT_SEASON
        params = {"league": league_id, "season": season}
        data = self._get("fixtures", params)
        return data.get("response", [])


def parse_lineups(lineups_data: list) -> dict:
    """Parses lineups into {player_id: 'starter'|'bench'} mapping."""
    result = {}
    for team_data in lineups_data:
        for p in team_data.get("startXI", []):
            pid = p.get("player", {}).get("id")
            if pid:
                result[pid] = "starter"
        for p in team_data.get("substitutes", []):
            pid = p.get("player", {}).get("id")
            if pid:
                result[pid] = "bench"
    return result


def parse_fixture(fixture_data: dict) -> dict:
    fixture = fixture_data.get("fixture", {})
    league = fixture_data.get("league", {})
    teams = fixture_data.get("teams", {})
    goals = fixture_data.get("goals", {})
    score = fixture_data.get("score", {})

    status_map = {
        "NS": "scheduled",
        "FT": "finished",
        "AET": "finished",
        "PEN": "finished",
        "PST": "postponed",
        "CANC": "postponed",
        "1H": "live",
        "2H": "live",
        "HT": "live",
    }
    raw_status = fixture.get("status", {}).get("short", "")

    halftime = score.get("halftime", {})

    return {
        "api_fixture_id": fixture.get("id"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "season": league.get("season"),
        "match_date": fixture.get("date"),
        "home_team": teams.get("home", {}).get("name"),
        "away_team": teams.get("away", {}).get("name"),
        "home_team_id": teams.get("home", {}).get("id"),
        "away_team_id": teams.get("away", {}).get("id"),
        "home_score": goals.get("home"),
        "away_score": goals.get("away"),
        "home_ht_score": halftime.get("home"),
        "away_ht_score": halftime.get("away"),
        "status": status_map.get(raw_status, "unknown"),
        "home_corners": None,
        "away_corners": None,
        "home_shots_on_target": None,
        "away_shots_on_target": None,
        "home_possession": None,
        "away_possession": None,
        "venue": fixture.get("venue", {}).get("name"),
    }


def parse_fixture_statistics(stats_data: list, match_data: dict) -> dict:
    """Extrae stats de un fixture y las agrega al dict del match."""
    for team_stats in stats_data:
        team_id = team_stats.get("team", {}).get("id")
        statistics = {s["type"]: s["value"] for s in team_stats.get("statistics", [])}

        is_home = team_id == match_data.get("home_team_id")
        prefix = "home" if is_home else "away"

        corners = statistics.get("Corner Kicks")
        shots_on = statistics.get("Shots on Goal")
        possession = statistics.get("Ball Possession")

        if corners is not None:
            match_data[f"{prefix}_corners"] = int(corners) if corners else 0
        if shots_on is not None:
            match_data[f"{prefix}_shots_on_target"] = int(shots_on) if shots_on else 0
        if possession is not None:
            poss_str = str(possession).replace("%", "")
            try:
                match_data[f"{prefix}_possession"] = float(poss_str)
            except ValueError:
                pass

    return match_data


def parse_odds(odds_data: list, preferred_bookmaker: str = "Bet365") -> dict:
    """Parse odds response into a flat dict of odds values.

    Tries preferred bookmaker first, falls back to first available.
    Returns dict with keys: odds_home_win, odds_draw, odds_away_win,
    odds_over15, odds_under15, odds_over25, odds_under25,
    odds_over35, odds_under35, odds_btts_yes, odds_btts_no.
    """
    if not odds_data:
        return {}

    bookmakers = odds_data[0].get("bookmakers", []) if odds_data else []
    if not bookmakers:
        return {}

    # Find preferred bookmaker, fallback to first
    bk = None
    for b in bookmakers:
        if b.get("name") == preferred_bookmaker:
            bk = b
            break
    if bk is None:
        bk = bookmakers[0]

    bets_by_name = {}
    for bet in bk.get("bets", []):
        bets_by_name[bet.get("name")] = {
            v.get("value"): float(v.get("odd", 0))
            for v in bet.get("values", [])
        }

    result = {"bookmaker": bk.get("name", "")}

    # 1X2
    mw = bets_by_name.get("Match Winner", {})
    if mw:
        result["odds_home_win"] = mw.get("Home", 0)
        result["odds_draw"] = mw.get("Draw", 0)
        result["odds_away_win"] = mw.get("Away", 0)

    # Over/Under goals
    ou = bets_by_name.get("Goals Over/Under", {})
    if ou:
        result["odds_over15"] = ou.get("Over 1.5", 0)
        result["odds_under15"] = ou.get("Under 1.5", 0)
        result["odds_over25"] = ou.get("Over 2.5", 0)
        result["odds_under25"] = ou.get("Under 2.5", 0)
        result["odds_over35"] = ou.get("Over 3.5", 0)
        result["odds_under35"] = ou.get("Under 3.5", 0)

    # BTTS
    btts = bets_by_name.get("Both Teams Score", {})
    if btts:
        result["odds_btts_yes"] = btts.get("Yes", 0)
        result["odds_btts_no"] = btts.get("No", 0)

    return result


def parse_player_fixture_stats(players_data: list) -> list:
    """Extrae stats de jugadores de un fixture."""
    result = []
    for team_data in players_data:
        team = team_data.get("team", {})
        for player_data in team_data.get("players", []):
            player = player_data.get("player", {})
            stats_list = player_data.get("statistics", [{}])
            stats = stats_list[0] if stats_list else {}
            shots = stats.get("shots", {}) or {}
            games = stats.get("games", {}) or {}

            goals = stats.get("goals", {}) or {}

            result.append({
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "shots_on_target": shots.get("on") or 0,
                "shots_total": shots.get("total") or 0,
                "goals": goals.get("total") or 0,
                "minutes_played": games.get("minutes") or 0,
            })
    return result
