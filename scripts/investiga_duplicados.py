"""
Investigar causa raíz de las predicciones duplicadas de NAC Breda vs Heerenveen.
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

print("\n=== INVESTIGACIÓN DUPLICADOS NAC BREDA vs HEERENVEEN ===\n")

# 1. Detalle completo de las predicciones duplicadas
rows = q("""
    SELECT p.id, p.match_id, p.market_type, p.prediction,
           p.prediction_date::text, p.created_at::text,
           p.data_source, p.model_version, p.result,
           p.probability, p.player_id
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    WHERE m.home_team = 'NAC Breda' AND m.away_team = 'Heerenveen'
    ORDER BY p.market_type, p.id
""")

print(f"  Total predicciones para este partido: {len(rows)}\n")
for r in rows:
    print(f"  id={r['id']} | market={r['market_type']} | created={r['created_at']}")
    print(f"    prediction_date={r['prediction_date']} | source={r['data_source']}")
    print(f"    model={r['model_version']} | prob={r['probability']:.3f}")
    print(f"    prediction: {r['prediction']}")
    print()

# 2. ¿Las dos predicciones fueron creadas al mismo tiempo o en momentos distintos?
print("  --- ANÁLISIS DE TIMESTAMPS ---")
rows2 = q("""
    SELECT p.market_type,
           MIN(p.created_at)::text as primera,
           MAX(p.created_at)::text as segunda,
           EXTRACT(EPOCH FROM (MAX(p.created_at) - MIN(p.created_at))) as segundos_entre
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    WHERE m.home_team = 'NAC Breda' AND m.away_team = 'Heerenveen'
    GROUP BY p.market_type
    HAVING COUNT(*) > 1
""")
for r in rows2:
    seg = float(r['segundos_entre'])
    print(f"  {r['market_type']}: primera={r['primera']} | segunda={r['segunda']}")
    if seg < 5:
        print(f"    -> {seg:.1f} seg de diferencia = MISMO CICLO (doble insert en un run)")
    elif seg < 120:
        print(f"    -> {seg:.1f} seg = dos runs casi simultáneos (race condition)")
    else:
        minutos = seg / 60
        print(f"    -> {minutos:.1f} min = dos ejecuciones separadas del flujo")

# 3. ¿Cuántos prediction_date distintos tienen?
rows3 = q("""
    SELECT DISTINCT p.prediction_date::text, p.created_at::text,
           p.data_source, p.model_version
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    WHERE m.home_team = 'NAC Breda' AND m.away_team = 'Heerenveen'
    ORDER BY p.created_at
""")
print("\n  --- FECHAS DISTINTAS ---")
for r in rows3:
    print(f"  pred_date={r['prediction_date']} | created={r['created_at']} | source={r['data_source']}")

print()
