"""
FIX D: Odds leakage check usando captured_at.
Ejecutar: python scripts/fix_d_check.py
"""
import sys
import os
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


print("\n=== FIX D: ODDS LEAKAGE CHECK (captured_at vs match_date) ===\n")

# Odds capturadas DESPUES del partido (todos los status)
leakage_all = q("""
    SELECT COUNT(*) AS n
    FROM odds_history o
    JOIN matches m ON o.match_id = m.id
    WHERE o.captured_at > m.match_date
""")
n_all = leakage_all[0]["n"]
print(f"  Odds capturadas DESPUES del partido (todos):    {n_all}")

# Odds capturadas DESPUES del partido (solo finished)
leakage_fin = q("""
    SELECT COUNT(*) AS n
    FROM odds_history o
    JOIN matches m ON o.match_id = m.id
    WHERE o.captured_at > m.match_date
      AND m.status = 'finished'
""")
n_fin = leakage_fin[0]["n"]
print(f"  Odds capturadas DESPUES del partido (finished): {n_fin}")

# Odds correctas (antes del partido)
ok = q("""
    SELECT COUNT(*) AS n
    FROM odds_history o
    JOIN matches m ON o.match_id = m.id
    WHERE o.captured_at < m.match_date
""")
print(f"  Odds capturadas ANTES del partido (correcto):   {ok[0]['n']}")

# Top casos de leakage
if n_all > 0:
    top = q("""
        SELECT m.home_team, m.away_team,
               o.captured_at::text AS odds_captured,
               m.match_date::text AS match_date,
               ROUND(EXTRACT(EPOCH FROM (o.captured_at - m.match_date))/3600) AS horas_despues
        FROM odds_history o
        JOIN matches m ON o.match_id = m.id
        WHERE o.captured_at > m.match_date
        ORDER BY horas_despues DESC
        LIMIT 5
    """)
    print("\n  Top casos de leakage:")
    for r in top:
        ht = r["home_team"]
        at = r["away_team"]
        oc = r["odds_captured"]
        md = r["match_date"]
        hd = r["horas_despues"]
        print(f"    {ht} vs {at} | odds={oc} | match={md} | +{hd}h despues")
    print()
    if n_fin > 0:
        print("  !! LEAKAGE ACTIVO EN DATOS DE ENTRENAMIENTO !!")
        print("  -> Se deben excluir columnas odds_* de MATCH_FEATURE_NAMES")
        print("     en trainer.py y reentrenar el modelo 1x2.")
    else:
        print("  INFO: Leakage solo en partidos no-finished (scheduled/live).")
        print("  -> No afecta entrenamiento (trainer.py filtra WHERE status=finished).")
        print("  -> SIN IMPACTO en el modelo.")
else:
    print("\n  OK: Sin leakage detectado. Las odds se guardan antes del partido.")
    print("  -> No es necesario modificar MATCH_FEATURE_NAMES.")

print()
