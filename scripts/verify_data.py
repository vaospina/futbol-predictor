"""
Verificacion de datos historicos en Supabase.
Ejecutar: python scripts/verify_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine, text

engine = create_engine(os.getenv("DATABASE_URL"))

def q(sql):
    with engine.connect() as conn:
        r = conn.execute(text(sql))
        cols = list(r.keys())
        return [dict(zip(cols, row)) for row in r.fetchall()]

SEP = "=" * 55
print(SEP)
print("  VERIFICACION DATOS HISTORICOS")
print(SEP)

# 1. Matches con resultado
print("\n[ 1 ] MATCHES con home_score IS NOT NULL")
r = q("""SELECT COUNT(*) as total_matches,
                MIN(match_date)::text as desde,
                MAX(match_date)::text as hasta
         FROM matches WHERE home_score IS NOT NULL""")
row = r[0]
print(f"  total_matches : {row['total_matches']}")
print(f"  desde         : {row['desde']}")
print(f"  hasta         : {row['hasta']}")

# 2. Predicciones evaluadas (via JOIN)
print("\n[ 2 ] PREDICTIONS evaluadas (result IS NOT NULL)")
r = q("""SELECT COUNT(*) as predictions_evaluadas,
                MIN(m.match_date)::text as desde,
                MAX(m.match_date)::text as hasta
         FROM predictions p
         JOIN matches m ON p.match_id = m.id
         WHERE p.result IS NOT NULL""")
row = r[0]
print(f"  predictions_evaluadas : {row['predictions_evaluadas']}")
print(f"  desde                 : {row['desde']}")
print(f"  hasta                 : {row['hasta']}")

r2 = q("""SELECT result, COUNT(*) as n
          FROM predictions WHERE result IS NOT NULL
          GROUP BY result ORDER BY n DESC""")
print("  Desglose por result:")
for row in r2:
    print(f"    {str(row['result']):10s}: {row['n']}")

# 3. Player stats
print("\n[ 3 ] PLAYER_STATS")
r = q("""SELECT COUNT(*) as total,
                MIN(match_date)::text as desde,
                MAX(match_date)::text as hasta
         FROM player_stats""")
row = r[0]
print(f"  player_stats_total : {row['total']}")
print(f"  desde              : {row['desde']}")
print(f"  hasta              : {row['hasta']}")

# 4. Corners en matches
print("\n[ 4 ] COLUMNAS corners en tabla matches")
cols = q("""SELECT column_name FROM information_schema.columns
            WHERE table_name='matches' AND column_name LIKE '%corner%'""")
if cols:
    for c in cols:
        print(f"  columna: {c['column_name']}")
    r = q("SELECT COUNT(*) as n FROM matches WHERE home_corners IS NOT NULL")
    print(f"  Filas con home_corners IS NOT NULL: {r[0]['n']}")
    r2 = q("""SELECT ROUND(AVG(home_corners+away_corners)::numeric,2) as avg_total,
                     MIN(home_corners+away_corners) as min_c,
                     MAX(home_corners+away_corners) as max_c
              FROM matches WHERE home_corners IS NOT NULL""")
    row = r2[0]
    print(f"  avg corners totales: {row['avg_total']}")
    print(f"  min / max          : {row['min_c']} / {row['max_c']}")
else:
    print("  !! No se encontraron columnas de corners en matches")

# 5. Predicciones duplicadas
print("\n[ 5 ] PREDICCIONES DUPLICADAS (mismo fixture+market+prediction)")
r = q("""SELECT p.prediction, p.market_type,
                m.home_team, m.away_team, m.api_fixture_id,
                COUNT(*) as n
         FROM predictions p
         JOIN matches m ON p.match_id = m.id
         GROUP BY p.prediction, p.market_type,
                  m.home_team, m.away_team, m.api_fixture_id
         HAVING COUNT(*) > 1
         ORDER BY n DESC
         LIMIT 10""")
if r:
    print(f"  !! {len(r)} grupos con duplicados:")
    for row in r:
        pred = str(row['prediction'])[:50]
        print(f"    x{row['n']} | {row['home_team']} vs {row['away_team']} "
              f"| {row['market_type']} | {pred}")
else:
    print("  OK: Sin duplicados en el historial total")

# 6. Constraints en predictions
print("\n[ 6 ] CONSTRAINTS en tabla predictions")
r = q("""SELECT constraint_name, constraint_type
         FROM information_schema.table_constraints
         WHERE table_name = 'predictions'""")
for row in r:
    print(f"  {str(row['constraint_type']):20s}: {row['constraint_name']}")

# 7. Columnas de predictions
print("\n[ 7 ] Columnas tabla predictions")
r = q("""SELECT column_name, data_type
         FROM information_schema.columns
         WHERE table_name='predictions'
         ORDER BY ordinal_position""")
for row in r:
    print(f"  {str(row['column_name']):25s} {row['data_type']}")

print(SEP)
