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

    def get_odds(self, fixture_id: int, bookmaker: int = 6):
        params = {"fixture": fixture_id, "bookmaker": bookmaker}
        data = self._get("odds", params)
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

            result.append({
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "shots_on_target": shots.get("on") or 0,
                "shots_total": shots.get("total") or 0,
                "minutes_played": games.get("minutes") or 0,
            })
    return result
