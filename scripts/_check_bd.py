import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from db.models import fetch_all, fetch_one

print("=== MATCHES EN BD POR LIGA Y TEMPORADA ===")
rows = fetch_all("""
    SELECT league_id, league_name, season,
           COUNT(*) as total,
           COUNT(CASE WHEN status='finished' THEN 1 END) as finished,
           COUNT(CASE WHEN status='finished' AND home_corners IS NOT NULL THEN 1 END) as con_stats,
           MIN(DATE(match_date)) as primera_fecha,
           MAX(DATE(match_date)) as ultima_fecha
    FROM matches
    GROUP BY league_id, league_name, season
    ORDER BY season DESC, league_id
""")

print(f"{'Liga':30s} {'Temp':5s} {'Total':6s} {'Fin':5s} {'Stats':6s} {'Desde':11s} {'Hasta':11s}")
print("-" * 80)
for r in rows:
    name = (r['league_name'] or '?')[:30]
    print(f"{name:30s} {str(r['season']):5s} {r['total']:6d} {r['finished']:5d} {r['con_stats']:6d} {str(r['primera_fecha']):11s} {str(r['ultima_fecha']):11s}")

print()
r = fetch_one("SELECT COUNT(*) as n, COUNT(CASE WHEN status='finished' THEN 1 END) as f FROM matches")
print(f"TOTAL GLOBAL: {r['n']} matches, {r['f']} finished")

print()
print("=== GAPS: LIGAS CONFIGURADAS SIN DATOS ===")
configured = {
    39: "Premier League", 135: "Serie A", 78: "Bundesliga", 140: "La Liga",
    61: "Ligue 1", 88: "Eredivisie", 71: "Brasileirao Serie A", 262: "Liga MX",
    128: "Argentine Primera", 2: "Champions League", 3: "Europa League", 13: "Copa Libertadores"
}
# Map (league_id, season) -> con_stats
db_map = {(r['league_id'], r['season']): r['con_stats'] for r in rows}

for season in [2024, 2023]:
    print(f"\n  Temporada {season}:")
    for lid, name in configured.items():
        stats_count = db_map.get((lid, season), None)
        if stats_count is None:
            print(f"    !! FALTA completamente: {name} (id={lid})")
        elif stats_count == 0:
            total = db_map.get((lid, season), 0)
            print(f"    ?? SIN stats: {name} (id={lid}) — {total} matches pero 0 con corners/shots")
        else:
            finished = next((r['finished'] for r in rows if r['league_id']==lid and r['season']==season), 0)
            print(f"    OK {name}: {stats_count} con stats / {finished} finished")
