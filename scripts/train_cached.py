"""
Entrena los 3 modelos con un cache en memoria de matches y player_stats
para evitar miles de queries remotas a Supabase durante feature engineering.

Uso:
    python -m scripts.train_cached
"""
import json
from collections import defaultdict
from datetime import date, datetime

from db.models import fetch_all
from db import models as db_models
from models import feature_engineering as fe
from utils.logger import get_logger

logger = get_logger(__name__)


def _to_date(d):
    if d is None:
        return None
    if isinstance(d, str):
        return date.fromisoformat(d[:10])
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return None


def build_caches():
    logger.info("Cargando matches en memoria...")
    matches = fetch_all(
        "SELECT * FROM matches WHERE status='finished' ORDER BY match_date ASC"
    )
    logger.info(f"  {len(matches)} matches cargados")

    # Normalizar match_date a date
    for m in matches:
        m["_date"] = _to_date(m.get("match_date"))

    # Índices por equipo (home, away, all) ordenados asc por fecha
    home_idx = defaultdict(list)  # team_id -> list of matches where team is home
    away_idx = defaultdict(list)  # team_id -> list of matches where team is away
    all_idx = defaultdict(list)   # team_id -> list of matches where team plays (home or away)

    for m in matches:
        h = m.get("home_team_id")
        a = m.get("away_team_id")
        if h:
            home_idx[h].append(m)
            all_idx[h].append(m)
        if a:
            away_idx[a].append(m)
            all_idx[a].append(m)

    logger.info("Cargando player_stats en memoria...")
    players = fetch_all(
        """SELECT ps.*, m.home_team_id, m.away_team_id, m.league_id AS m_league_id,
                  m.home_corners, m.away_corners,
                  m.home_shots_on_target AS match_home_sot,
                  m.away_shots_on_target AS match_away_sot,
                  m.match_date AS m_match_date,
                  m.home_team, m.away_team
           FROM player_stats ps
           JOIN matches m ON ps.match_id = m.id
           WHERE ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
           ORDER BY ps.match_date ASC"""
    )
    logger.info(f"  {len(players)} player_stats cargados")

    player_idx = defaultdict(list)
    for p in players:
        p["_date"] = _to_date(p.get("match_date"))
        player_idx[p["player_id"]].append(p)

    # odds vacío por defecto (no hicimos backfill de odds)
    return {
        "matches": matches,
        "home_idx": home_idx,
        "away_idx": away_idx,
        "all_idx": all_idx,
        "players": players,
        "player_idx": player_idx,
    }


def install_patches(cache):
    home_idx = cache["home_idx"]
    away_idx = cache["away_idx"]
    all_idx = cache["all_idx"]
    player_idx = cache["player_idx"]

    def get_team_home_matches(team_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = home_idx.get(team_id, [])
        filtered = [m for m in lst if m["_date"] and m["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_team_away_matches(team_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = away_idx.get(team_id, [])
        filtered = [m for m in lst if m["_date"] and m["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_team_last_matches(team_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = all_idx.get(team_id, [])
        filtered = [m for m in lst if m["_date"] and m["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_h2h_matches(t1, t2, n=5, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = all_idx.get(t1, [])
        both = [
            m for m in lst
            if m["_date"] and m["_date"] < before
            and (
                (m.get("home_team_id") == t1 and m.get("away_team_id") == t2)
                or (m.get("home_team_id") == t2 and m.get("away_team_id") == t1)
            )
        ]
        return list(reversed(both[-n:]))

    def get_player_recent_stats(player_id, n=10, before_date=None):
        before = _to_date(before_date) or date.today()
        lst = player_idx.get(player_id, [])
        filtered = [p for p in lst if p["_date"] and p["_date"] < before]
        return list(reversed(filtered[-n:]))

    def get_match_odds_summary(match_id):
        return {}

    # Patch db.models
    db_models.get_team_home_matches = get_team_home_matches
    db_models.get_team_away_matches = get_team_away_matches
    db_models.get_team_last_matches = get_team_last_matches
    db_models.get_h2h_matches = get_h2h_matches
    db_models.get_player_recent_stats = get_player_recent_stats

    # feature_engineering importó las funciones directamente; re-bindear ahí
    fe.get_team_home_matches = get_team_home_matches
    fe.get_team_away_matches = get_team_away_matches
    fe.get_team_last_matches = get_team_last_matches
    fe.get_h2h_matches = get_h2h_matches
    fe.get_player_recent_stats = get_player_recent_stats
    fe.get_match_odds_summary = get_match_odds_summary

    # models.trainer también importó fetch_all, feature builders, etc.
    # No necesitamos patchear nada más porque los imports referenciados por nombre.
    logger.info("Patches instalados (cache en memoria activo)")


def main():
    cache = build_caches()
    install_patches(cache)

    # Import después de patch para asegurar que trainer use las nuevas funciones
    from models.trainer import train_all_models
    logger.info("Iniciando train_all_models con cache...")
    version = datetime.now().strftime("v%Y%m%d_%H%M")
    results = train_all_models(version)
    logger.info("=== RESULTS ===")
    logger.info(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
