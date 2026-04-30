"""
Backfill dirigido para las ligas nuevas (La Liga, Ligue 1, Eredivisie,
Liga MX, Argentine Primera, Europa League) — temporadas 2024 y 2023.

Respeta el limite de 7500 req/dia de API-Football Pro.
Reserva 300 requests al final para las predicciones diarias del scheduler.

Uso:
    python -m scripts.backfill_new_leagues
    python -m scripts.backfill_new_leagues --only-season 2024
    python -m scripts.backfill_new_leagues --dry-run
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from datetime import date
from data.api_football import ApiFootballClient, parse_fixture, parse_fixture_statistics, parse_player_fixture_stats
from db.models import upsert_match, batch_upsert_player_stats, fetch_all, fetch_one
from db.migrations import run_migrations
from utils.logger import get_logger

logger = get_logger(__name__)

# Limite seguro: dejar 300 requests para predicciones diarias
DAILY_LIMIT = 7500
SAFETY_BUFFER = 300
MAX_REQUESTS = DAILY_LIMIT - SAFETY_BUFFER  # 7200

RATE_LIMIT_SLEEP = 0.15  # 400 req/min, dentro del limite Pro (450/min)

# Ligas nuevas a backfillear (las que faltan en BD)
NEW_LEAGUES = {
    140: {"name": "La Liga",           "country": "Espana"},
    61:  {"name": "Ligue 1",           "country": "Francia"},
    88:  {"name": "Eredivisie",        "country": "Holanda"},
    3:   {"name": "UEFA Europa League","country": "Europa"},
    262: {"name": "Liga MX",           "country": "Mexico"},
    128: {"name": "Argentine Primera", "country": "Argentina"},
}

# Rangos de fechas por tipo de liga y temporada
def get_date_range(league_id: int, season: int):
    latam = {262, 128}   # temporada = año calendario
    if league_id in latam:
        return date(season, 1, 1), date(season, 12, 31)
    # Europeas: temporada empieza en julio del año indicado
    return date(season, 7, 1), date(season + 1, 6, 30)


def backfill_league_season(api, league_id, league_info, season, existing_complete, dry_run=False):
    from_date, to_date = get_date_range(league_id, season)
    name = league_info["name"]

    logger.info(f"--- {name} ({league_id}) season={season} | {from_date} -> {to_date} ---")

    if dry_run:
        logger.info(f"  [DRY RUN] Se saltaria esta liga/temporada")
        return {"fixtures": 0, "finished": 0, "stats": 0, "players": 0, "api_calls": 1}

    fixtures = api.get_fixtures_by_date_range(league_id, from_date, to_date, season=season)
    time.sleep(RATE_LIMIT_SLEEP)

    finished = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") in
                ("FT", "AET", "PEN")]
    new_finished = [f for f in finished
                    if f.get("fixture", {}).get("id") not in existing_complete]

    logger.info(f"  Fixtures API: {len(fixtures)} total | {len(finished)} finished | "
                f"{len(new_finished)} nuevos (sin stats)")

    stats_ok = 0
    players_saved = 0

    for fixture_data in new_finished:
        fixture_id = fixture_data.get("fixture", {}).get("id")
        if not fixture_id:
            continue

        if api.requests_today >= MAX_REQUESTS:
            logger.warning(f"  Limite de seguridad alcanzado ({MAX_REQUESTS} req). Deteniendo.")
            break

        try:
            match_data = parse_fixture(fixture_data)

            # Stats del partido
            time.sleep(RATE_LIMIT_SLEEP)
            stats = api.get_fixture_statistics(fixture_id)
            if stats:
                parse_fixture_statistics(stats, match_data)
                stats_ok += 1

            match_id = upsert_match(match_data)
            if not match_id:
                continue

            existing_complete.add(fixture_id)

            # Player stats
            time.sleep(RATE_LIMIT_SLEEP)
            players_raw = api.get_fixture_player_stats(fixture_id)
            players = parse_player_fixture_stats(players_raw)

            match_date_val = (
                match_data["match_date"][:10]
                if isinstance(match_data["match_date"], str)
                else match_data["match_date"]
            )
            batch = []
            for p in players:
                if not p.get("player_id") or (p.get("minutes_played") or 0) < 1:
                    continue
                batch.append({
                    "player_id":      p["player_id"],
                    "player_name":    p["player_name"],
                    "team_id":        p["team_id"],
                    "team_name":      p["team_name"],
                    "league_id":      league_id,
                    "season":         season,
                    "match_id":       match_id,
                    "match_date":     match_date_val,
                    "shots_on_target":p["shots_on_target"],
                    "shots_total":    p["shots_total"],
                    "goals":          p.get("goals", 0),
                    "minutes_played": p["minutes_played"],
                })
            if batch:
                batch_upsert_player_stats(batch)
                players_saved += len(batch)

        except Exception as e:
            logger.error(f"  Error fixture {fixture_id}: {e}")
            continue

    logger.info(f"  {name} {season}: {stats_ok} stats | {players_saved} player rows | "
                f"{api.requests_today} req totales del dia")

    return {
        "fixtures": len(fixtures),
        "finished": len(new_finished),
        "stats": stats_ok,
        "players": players_saved,
        "api_calls": api.requests_today,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-season", type=int, choices=[2023, 2024, 2025],
                        help="Procesar solo esta temporada")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el plan sin llamar la API (excepto check de estado)")
    args = parser.parse_args()

    run_migrations()

    api = ApiFootballClient()

    # Verificar estado y cuota disponible
    status = api.check_status()

    if status.get("quota_exhausted") and not args.dry_run:
        logger.error("Cuota diaria de API-Football agotada. Abortando backfill.")
        sys.exit(1)

    reqs = status.get("requests", {})
    used_today = reqs.get("current", 0) or 0
    limit = reqs.get("limit_day", 7500) or 7500
    available = limit - used_today

    logger.info("=== BACKFILL LIGAS NUEVAS ===")
    logger.info(f"  Requests usados hoy: {used_today} / {limit}")
    logger.info(f"  Disponibles:         {available}")
    logger.info(f"  Limite seguro:       {MAX_REQUESTS} (reserva {SAFETY_BUFFER} para predicciones)")

    if available < 500 and not args.dry_run:
        logger.error(f"Solo {available} requests disponibles. Abortando para no agotar la cuota.")
        sys.exit(1)

    # Pre-cargar fixtures ya completos en BD
    rows = fetch_all(
        "SELECT api_fixture_id FROM matches WHERE status='finished' AND home_corners IS NOT NULL"
    )
    existing_complete = {r["api_fixture_id"] for r in rows}
    logger.info(f"  Fixtures ya completos en BD: {len(existing_complete)}")

    # Orden de prioridad: 2025 (mas reciente), luego 2024, 2023
    seasons = [2025, 2024, 2023]
    if args.only_season:
        seasons = [args.only_season]

    grand_total = {"fixtures": 0, "finished": 0, "stats": 0, "players": 0}

    for season in seasons:
        logger.info(f"\n{'='*50}")
        logger.info(f"=== TEMPORADA {season} ===")
        logger.info(f"{'='*50}")

        for league_id, league_info in NEW_LEAGUES.items():
            if api.requests_today >= MAX_REQUESTS:
                logger.warning(f"Limite {MAX_REQUESTS} alcanzado. Deteniendo backfill.")
                break

            remaining = MAX_REQUESTS - api.requests_today
            logger.info(f"\nRequests restantes del presupuesto: {remaining}")

            result = backfill_league_season(
                api, league_id, league_info, season,
                existing_complete, dry_run=args.dry_run
            )
            for k in grand_total:
                grand_total[k] += result.get(k, 0)

        else:
            continue
        break  # limite alcanzado, salir del loop de temporadas

    logger.info("\n=== BACKFILL COMPLETADO ===")
    logger.info(f"  Fixtures procesados: {grand_total['fixtures']}")
    logger.info(f"  Nuevos finished:     {grand_total['finished']}")
    logger.info(f"  Con stats:           {grand_total['stats']}")
    logger.info(f"  Player stats rows:   {grand_total['players']}")
    logger.info(f"  API calls totales:   {api.requests_today}")
    logger.info(f"  Requests restantes:  {MAX_REQUESTS - api.requests_today} (de presupuesto {MAX_REQUESTS})")

    # Resumen final de BD
    row = fetch_one("SELECT COUNT(*) AS n FROM matches WHERE status='finished'")
    logger.info(f"  Total matches finished en BD: {row['n'] if row else '?'}")


if __name__ == "__main__":
    main()
