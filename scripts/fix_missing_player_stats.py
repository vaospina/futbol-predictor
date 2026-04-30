"""
Reparación puntual: descarga player_stats para matches que tienen
corners guardados pero no tienen entradas en player_stats.
(Ocurre cuando batch_upsert_player_stats falló por columna faltante)

Uso:
    python -m scripts.fix_missing_player_stats
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from data.api_football import ApiFootballClient, parse_player_fixture_stats
from db.models import fetch_all, batch_upsert_player_stats
from utils.logger import get_logger

logger = get_logger(__name__)
SLEEP = 0.15


def main():
    api = ApiFootballClient()

    # Verificar cuota
    status = api.check_status()
    reqs = status.get("requests", {})
    used = reqs.get("current", 0) or 0
    limit = reqs.get("limit_day", 7500) or 7500
    available = limit - used
    logger.info(f"Requests usados: {used} / {limit} ({available} disponibles)")

    # Matches con corners pero sin player_stats
    missing = fetch_all("""
        SELECT m.id as match_id, m.api_fixture_id, m.league_id, m.league_name,
               m.season, m.match_date, m.home_team, m.away_team
        FROM matches m
        WHERE m.status = 'finished'
          AND m.home_corners IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM player_stats ps WHERE ps.match_id = m.id
          )
        ORDER BY m.season DESC, m.match_date
    """)

    logger.info(f"Matches a reparar: {len(missing)}")

    fixed = 0
    skipped = 0

    for m in missing:
        if api.requests_today >= available - 100:
            logger.warning("Cuota casi agotada, deteniendo.")
            break

        fixture_id = m["api_fixture_id"]
        match_id = m["match_id"]
        league_id = m["league_id"]
        season = m["season"]

        try:
            time.sleep(SLEEP)
            players_raw = api.get_fixture_player_stats(fixture_id)
            players = parse_player_fixture_stats(players_raw)

            match_date_val = (
                m["match_date"].isoformat() if hasattr(m["match_date"], "isoformat")
                else str(m["match_date"])[:10]
            )

            batch = []
            for p in players:
                if not p.get("player_id") or (p.get("minutes_played") or 0) < 1:
                    continue
                batch.append({
                    "player_id":       p["player_id"],
                    "player_name":     p["player_name"],
                    "team_id":         p["team_id"],
                    "team_name":       p["team_name"],
                    "league_id":       league_id,
                    "season":          season,
                    "match_id":        match_id,
                    "match_date":      match_date_val,
                    "shots_on_target": p["shots_on_target"],
                    "shots_total":     p["shots_total"],
                    "goals":           p.get("goals", 0),
                    "minutes_played":  p["minutes_played"],
                })

            if batch:
                batch_upsert_player_stats(batch)
                fixed += 1
                logger.info(f"  [{fixed}] Fixture {fixture_id} {m['home_team']} vs {m['away_team']}: "
                            f"{len(batch)} player rows guardados")
            else:
                skipped += 1
                logger.debug(f"  Fixture {fixture_id}: sin player data")

        except Exception as e:
            logger.error(f"  Error fixture {fixture_id}: {e}")
            continue

    logger.info(f"\n=== FIX COMPLETADO ===")
    logger.info(f"  Fixtures reparados:  {fixed}")
    logger.info(f"  Sin datos (API):     {skipped}")
    logger.info(f"  API calls usados:    {api.requests_today}")


if __name__ == "__main__":
    main()
