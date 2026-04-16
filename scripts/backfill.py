"""
Backfill histórico desde API-Football hacia Supabase.

Carga fixtures + estadísticas + player stats de todas las ligas configuradas
para un rango de fechas dado y una temporada específica.

Uso:
    python -m scripts.backfill                                    # 2025-08-01 -> hoy, season=CURRENT
    python -m scripts.backfill 2026-01-01 2026-04-14              # rango custom, season=CURRENT
    python -m scripts.backfill 2023-07-01 2024-07-31 2023         # rango + season explícita
"""
import sys
import time
from datetime import date, datetime

from data.api_football import (
    ApiFootballClient,
    parse_fixture,
    parse_fixture_statistics,
    parse_player_fixture_stats,
)
from db.models import upsert_match, upsert_player_stat, fetch_one, get_match_by_fixture_id
from db.migrations import run_migrations
from config.leagues import ALL_LEAGUES
from config.settings import CURRENT_SEASON
from utils.logger import get_logger

logger = get_logger(__name__)

RATE_LIMIT_SLEEP = 0.15  # ~400 req/min, dentro del límite Pro (450/min)


def backfill(from_date: date, to_date: date, season: int = None):
    season = season or CURRENT_SEASON
    api = ApiFootballClient()

    logger.info(f"=== BACKFILL {from_date} -> {to_date} (season={season}) ===")
    run_migrations()

    total_fixtures = 0
    total_finished = 0
    total_stats_ok = 0
    total_players = 0

    for league_id, league_info in ALL_LEAGUES.items():
        logger.info(f"--- Liga {league_id} {league_info['name']} ---")
        fixtures = api.get_fixtures_by_date_range(
            league_id, from_date, to_date, season=season
        )
        logger.info(f"  {len(fixtures)} fixtures recibidos")
        total_fixtures += len(fixtures)

        for fixture_data in fixtures:
            try:
                match_data = parse_fixture(fixture_data)
                fixture_id = match_data["api_fixture_id"]
                if not fixture_id:
                    continue

                is_finished = match_data["status"] == "finished"

                # Skip si ya está completo en BD (re-runs incrementales)
                existing = get_match_by_fixture_id(fixture_id)
                already_complete = (
                    existing
                    and existing.get("status") == "finished"
                    and existing.get("home_corners") is not None
                )
                if already_complete:
                    continue

                if is_finished:
                    # Stats del partido (corners, shots, posesión)
                    time.sleep(RATE_LIMIT_SLEEP)
                    stats = api.get_fixture_statistics(fixture_id)
                    if stats:
                        parse_fixture_statistics(stats, match_data)
                        total_stats_ok += 1

                match_id = upsert_match(match_data)
                if not match_id:
                    continue

                if is_finished:
                    total_finished += 1
                    # Player stats
                    time.sleep(RATE_LIMIT_SLEEP)
                    players_raw = api.get_fixture_player_stats(fixture_id)
                    players = parse_player_fixture_stats(players_raw)
                    for p in players:
                        if not p.get("player_id"):
                            continue
                        if (p.get("minutes_played") or 0) < 1:
                            continue
                        upsert_player_stat({
                            "player_id": p["player_id"],
                            "player_name": p["player_name"],
                            "team_id": p["team_id"],
                            "team_name": p["team_name"],
                            "league_id": league_id,
                            "season": season,
                            "match_id": match_id,
                            "match_date": match_data["match_date"][:10]
                                if isinstance(match_data["match_date"], str)
                                else match_data["match_date"],
                            "shots_on_target": p["shots_on_target"],
                            "shots_total": p["shots_total"],
                            "minutes_played": p["minutes_played"],
                        })
                        total_players += 1

            except Exception as e:
                logger.error(f"  Error fixture {fixture_data.get('fixture', {}).get('id')}: {e}")
                continue

        logger.info(
            f"  Liga {league_info['name']} OK. API calls acumuladas: {api.requests_today}"
        )

    logger.info("=== BACKFILL COMPLETADO ===")
    logger.info(f"  Fixtures totales:   {total_fixtures}")
    logger.info(f"  Fixtures finished:  {total_finished}")
    logger.info(f"  Con stats:          {total_stats_ok}")
    logger.info(f"  Player stats rows:  {total_players}")
    logger.info(f"  API calls totales:  {api.requests_today}")

    row = fetch_one("SELECT COUNT(*) AS n FROM matches WHERE status='finished'")
    logger.info(f"  Matches finished en DB: {row['n'] if row else '?'}")

    return {
        "fixtures": total_fixtures,
        "finished": total_finished,
        "stats": total_stats_ok,
        "players": total_players,
        "api_calls": api.requests_today,
    }


if __name__ == "__main__":
    season_arg = None
    if len(sys.argv) >= 3:
        from_d = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        to_d = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
        if len(sys.argv) >= 4:
            season_arg = int(sys.argv[3])
    else:
        from_d = date(2025, 8, 1)
        to_d = date.today()
    backfill(from_d, to_d, season=season_arg)
