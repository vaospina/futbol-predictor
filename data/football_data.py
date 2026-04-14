"""
Cliente Football-Data.org - Fuente secundaria/validacion.
Plan gratuito: 10 requests/minuto.
Cubre: Premier League, Serie A, Bundesliga, Champions League.
NO cubre: Brasileirao, Libertadores.
"""
import requests
import os
from utils.logger import get_logger

logger = get_logger(__name__)

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "")
BASE_URL = "https://api.football-data.org/v4"

# Mapeo de league IDs de API-Football a competicion codes de Football-Data
LEAGUE_MAP = {
    39: "PL",    # Premier League
    135: "SA",   # Serie A
    78: "BL1",   # Bundesliga
    2: "CL",     # Champions League
}


class FootballDataClient:
    def __init__(self):
        self.headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}

    def _get(self, endpoint: str) -> dict:
        if not FOOTBALL_DATA_TOKEN:
            logger.debug("Football-Data.org token no configurado, omitiendo")
            return {}
        url = f"{BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Football-Data error ({response.status_code})")
                return {}
        except Exception as e:
            logger.error(f"Error Football-Data: {e}")
            return {}

    def get_matches(self, competition_code: str, date_from: str, date_to: str):
        return self._get(
            f"competitions/{competition_code}/matches?dateFrom={date_from}&dateTo={date_to}"
        )

    def get_standings(self, competition_code: str):
        return self._get(f"competitions/{competition_code}/standings")
