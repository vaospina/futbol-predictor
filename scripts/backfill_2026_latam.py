"""
Backfill temporada 2026 para ligas LATAM y copas que usan año calendario:
  - Liga MX (262)
  - Argentine Primera (128)
  - Copa Libertadores (13)
  - Brasileirao Serie A (71)

Las ligas europeas (season 2025) ya tienen datos 2026 del backfill anterior.
Respeta el límite de 7,500 req/día con buffer de 300 para predicciones.

Uso:
    python -m scripts.backfill_2026_latam
    python -m scripts.backfill_2026_latam --dry-run
"""
import sys, os, time, argparse
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from data.api_football import ApiFootballClient, parse_fixture, parse_fixture_statistics, parse_player_fixture_stats
from db.models import upsert_match, batch_upsert_player_stats, fetch_all, fetch_one
from db.migrations import run_migrations
from utils.logger import get_logger

logger = get_logger(__name__)

DAILY_LIMIT   = 7500
SAFETY_BUFFER = 300
MAX_REQUESTS  = DAILY_LIMIT - SAFETY_BUFFER
SLEEP         = 0.15

SEASON_2026 = 2026
TODAY = date.today()

# Ligas que usan año calendario: season 2026 = ene-dic 2026
LATAM_LEAGUES_2026 = {
    262: {"name": "Liga MX",           "from": date(2026, 1, 1),  "to": TODAY},
    128: {"name": "Argentine Primera", "from": date(2026, 1, 1),  "to": TODAY},
    13:  {"name": "Copa Libertadores", "from": date(2026, 2, 1),  "to": TODAY},
    71:  {"name": "Brasileirao",       "from": date(2026, 4, 1),  "to": TODAY},
}

# Liga MX uses season=2025 for Clausura 2026 (Jan-Jun 2026)
# API-Football numbers the season by the Apertura year (July start)
LIGA_MX_CLAUSURA_2026 = {
    "league_id": 262,
    "api_season": 2025,
    "name": "Liga MX Clausura 2026",
    "from": date(2026, 1, 1),
    "to": TODAY,
}


def backfill_league(api, league_id, info, existing_complete, dry_run=False):
    from_date = info["from"]
    to_date   = info["to"]
    name      = info["name"]

    logger.info(f"--- {name} ({league_id}) season=2026 | {from_date} -> {to_date} ---")
    if dry_run:
        logger.info("  [DRY RUN]")
        return {"fixtures": 0, "new": 0, "stats": 0, "players": 0}

    fixtures = api.get_fixtures_by_date_range(league_id, from_date, to_date, season=SEASON_2026)
    time.sleep(SLEEP)

    finished = [
        f for f in fixtures
        if f.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
    ]
    new_finished = [
        f for f in finished
        if f.get("fixture", {}).get("id") not in existing_complete
    ]

    logger.info(f"  API: {len(fixtures)} total | {len(finished)} finished | {len(new_finished)} nuevos")

    stats_ok = 0
    players_saved = 0

    for fixture_data in new_finished:
        fixture_id = fixture_data.get("fixture", {}).get("id")
        if not fixture_id:
            continue

        if api.requests_today >= MAX_REQUESTS:
            logger.warning(f"  Límite {MAX_REQUESTS} alcanzado. Deteniendo.")
            break

        try:
            match_data = parse_fixture(fixture_data)

            time.sleep(SLEEP)
            stats = api.get_fixture_statistics(fixture_id)
            if stats:
                parse_fixture_statistics(stats, match_data)
                stats_ok += 1

            match_id = upsert_match(match_data)
            if not match_id:
                continue

            existing_complete.add(fixture_id)

            time.sleep(SLEEP)
            players_raw = api.get_fixture_player_stats(fixture_id)
            players = parse_player_fixture_stats(players_raw)

            match_date_val = (
                match_data["match_date"][:10]
                if isinstance(match_data["match_date"], str)
                else str(match_data["match_date"])
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
                    "season":          SEASON_2026,
                    "match_id":        match_id,
                    "match_date":      match_date_val,
                    "shots_on_target": p["shots_on_target"],
                    "shots_total":     p["shots_total"],
                    "goals":           p.get("goals", 0),
                    "minutes_played":  p["minutes_played"],
                })
            if batch:
                batch_upsert_player_stats(batch)
                players_saved += len(batch)

        except Exception as e:
            logger.error(f"  Error fixture {fixture_id}: {e}")
            continue

    logger.info(
        f"  {name} 2026: {stats_ok} stats | {players_saved} player rows | "
        f"{api.requests_today} req totales"
    )
    return {"fixtures": len(fixtures), "new": len(new_finished), "stats": stats_ok, "players": players_saved}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_migrations()
    api = ApiFootballClient()

    status = api.check_status()
    if status.get("quota_exhausted") and not args.dry_run:
        logger.error("Cuota diaria agotada. Abortando.")
        sys.exit(1)

    reqs = status.get("requests", {})
    used = reqs.get("current", 0) or 0
    limit = reqs.get("limit_day", 7500) or 7500
    available = limit - used

    logger.info("=== BACKFILL 2026 LATAM ===")
    logger.info(f"  Requests usados: {used} / {limit} | Disponibles: {available}")
    logger.info(f"  Presupuesto máx: {MAX_REQUESTS}")

    if available < 500 and not args.dry_run:
        logger.error(f"Solo {available} disponibles. Abortando.")
        sys.exit(1)

    # Pre-cargar fixtures ya completos (cualquier temporada)
    rows = fetch_all(
        "SELECT api_fixture_id FROM matches WHERE status='finished' AND home_corners IS NOT NULL"
    )
    existing_complete = {r["api_fixture_id"] for r in rows}
    logger.info(f"  Fixtures existentes en BD: {len(existing_complete)}")

    grand = {"fixtures": 0, "new": 0, "stats": 0, "players": 0}

    for league_id, info in LATAM_LEAGUES_2026.items():
        if api.requests_today >= MAX_REQUESTS:
            logger.warning("Límite alcanzado. Deteniendo.")
            break
        remaining = MAX_REQUESTS - api.requests_today
        logger.info(f"\nRequests restantes: {remaining}")
        result = backfill_league(api, league_id, info, existing_complete, dry_run=args.dry_run)
        for k in grand:
            grand[k] += result.get(k, 0)

    # Liga MX Clausura 2026: uses season=2025 on the API (Apertura/Clausura numbering)
    if api.requests_today < MAX_REQUESTS and not args.dry_run:
        mx = LIGA_MX_CLAUSURA_2026
        logger.info(f"\nRequests restantes: {MAX_REQUESTS - api.requests_today}")
        fixtures = api.get_fixtures_by_date_range(
            mx["league_id"], mx["from"], mx["to"], season=mx["api_season"]
        )
        import time as _time; _time.sleep(SLEEP)
        finished = [
            f for f in fixtures
            if f.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
            and f.get("fixture", {}).get("date", "")[:4] == "2026"
        ]
        new_finished = [f for f in finished if f.get("fixture", {}).get("id") not in existing_complete]
        logger.info(f"--- {mx['name']} (season=2025 API) | {len(fixtures)} total | "
                    f"{len(finished)} finished 2026 | {len(new_finished)} nuevos ---")
        for fixture_data in new_finished:
            fixture_id = fixture_data.get("fixture", {}).get("id")
            if not fixture_id or api.requests_today >= MAX_REQUESTS:
                break
            try:
                match_data = parse_fixture(fixture_data)
                _time.sleep(SLEEP)
                stats = api.get_fixture_statistics(fixture_id)
                if stats:
                    parse_fixture_statistics(stats, match_data)
                match_id = upsert_match(match_data)
                if not match_id:
                    continue
                existing_complete.add(fixture_id)
                _time.sleep(SLEEP)
                players_raw = api.get_fixture_player_stats(fixture_id)
                players = parse_player_fixture_stats(players_raw)
                match_date_val = (
                    match_data["match_date"][:10]
                    if isinstance(match_data["match_date"], str)
                    else str(match_data["match_date"])
                )
                batch = [
                    {
                        "player_id": p["player_id"], "player_name": p["player_name"],
                        "team_id": p["team_id"], "team_name": p["team_name"],
                        "league_id": 262, "season": 2026,
                        "match_id": match_id, "match_date": match_date_val,
                        "shots_on_target": p["shots_on_target"], "shots_total": p["shots_total"],
                        "goals": p.get("goals", 0), "minutes_played": p["minutes_played"],
                    }
                    for p in players
                    if p.get("player_id") and (p.get("minutes_played") or 0) >= 1
                ]
                if batch:
                    batch_upsert_player_stats(batch)
                    grand["players"] += len(batch)
                grand["new"] += 1
                grand["stats"] += 1
            except Exception as e:
                logger.error(f"  Error fixture {fixture_id}: {e}")
        logger.info(f"Liga MX Clausura 2026: procesados {grand['new']} | {api.requests_today} req totales")

    logger.info("\n=== BACKFILL 2026 COMPLETADO ===")
    logger.info(f"  Fixtures procesados: {grand['fixtures']}")
    logger.info(f"  Nuevos insertados:   {grand['new']}")
    logger.info(f"  Con stats:           {grand['stats']}")
    logger.info(f"  Player stats rows:   {grand['players']}")
    logger.info(f"  API calls usados:    {api.requests_today}")

    r = fetch_one("SELECT COUNT(*) as n FROM matches WHERE season=2026")
    logger.info(f"  Total matches season 2026 en BD: {r['n'] if r else '?'}")


if __name__ == "__main__":
    main()
