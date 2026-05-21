"""
Analisis del ciclo de reentrenamiento de shots.
Ejecutar: python scripts/analisis_retrain.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

engine = create_engine(os.getenv("DATABASE_URL"))
SEP = "=" * 62

def q(sql, params=None):
    with engine.connect() as conn:
        r = conn.execute(text(sql), params or {})
        cols = list(r.keys())
        return [dict(zip(cols, row)) for row in r.fetchall()]

def q1(sql, params=None):
    rows = q(sql, params)
    return rows[0] if rows else {}

# ── 1. DATOS DE ENTRENAMIENTO ────────────────────────────────────
print(f"\n{SEP}")
print("[ 1 ] DATOS DE ENTRENAMIENTO SHOTS")
print(SEP)

total = q1("""
    SELECT COUNT(*) as n
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
""")
print(f"Total registros para entrenamiento shots: {total['n']}")

by_league = q("""
    SELECT m.league_name, m.league_id, COUNT(*) as n,
           MIN(ps.match_date::date) as primera,
           MAX(ps.match_date::date) as ultima
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
    GROUP BY m.league_name, m.league_id
    ORDER BY n DESC
""")
print(f"\nDesglose por liga ({len(by_league)} ligas):")
print(f"  {'Liga':<35} {'Registros':>10}  {'Desde':<12}  {'Hasta'}")
for r in by_league:
    print(f"  {str(r['league_name']):<35} {r['n']:>10}  {str(r['primera'])[:10]:<12}  {str(r['ultima'])[:10]}")

lib = q1("""
    SELECT COUNT(*) as n, COUNT(DISTINCT ps.player_id) as players,
           COUNT(DISTINCT ps.match_id) as matches
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE m.league_id = 13
      AND ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
""")
print(f"\nCopa Libertadores (league_id=13): {lib['n']} registros | {lib['players']} jugadores | {lib['matches']} partidos")

# ── 2. MUESTRAS EVALUADAS ────────────────────────────────────────
print(f"\n{SEP}")
print("[ 2 ] SHOTS EVALUADOS (result IS NOT NULL)")
print(SEP)

evaluated = q1("""
    SELECT COUNT(*) as total,
           COUNT(CASE WHEN p.result = 'win' THEN 1 END) as wins,
           COUNT(CASE WHEN p.result = 'loss' THEN 1 END) as losses
    FROM predictions p
    WHERE p.market_type = 'player_shots'
      AND p.result IS NOT NULL
""")
print(f"Predicciones shots evaluadas: {evaluated['total']}")
if evaluated['total']:
    acc = evaluated['wins'] / evaluated['total'] * 100
    print(f"  Wins: {evaluated['wins']} | Losses: {evaluated['losses']} | Accuracy: {acc:.1f}%")
print(f"\nMinimo requerido actualmente en trainer.py: NINGUNO")
print(f"Minimo propuesto para shots: 500 muestras evaluadas")

# ── 3. DATA LEAKAGE ──────────────────────────────────────────────
print(f"\n{SEP}")
print("[ 3 ] DATA LEAKAGE CHECK")
print(SEP)

future_stats = q1("""
    SELECT COUNT(*) as n
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE ps.shots_on_target IS NOT NULL
      AND ps.minutes_played > 45
      AND m.match_date >= CURRENT_DATE
""")
print(f"Registros con match_date >= hoy en training set: {future_stats['n']}")

today_preds = q1("""
    SELECT COUNT(*) as n
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE m.match_date::date = CURRENT_DATE
      AND ps.shots_on_target IS NOT NULL
""")
print(f"Stats de jugadores de HOY en player_stats: {today_preds['n']}")
print()
print("Analisis de leakage en trainer.py:")
print("  Query trainer: WHERE shots_on_target IS NOT NULL AND minutes_played > 45")
print("  -> No filtra match_date < NOW() en el dataset global")
print("  -> build_player_features(before_date=match_date) bloquea el partido actual")
print("  -> Leakage del partido en curso: NO (before_date lo previene)")
print(f"  -> Leakage de stats futuras: {'SI - hay ' + str(future_stats['n']) + ' registros futuros' if future_stats['n'] > 0 else 'NO (sin stats futuras en BD)'}")

# ── 4. CROSS-VALIDATION ──────────────────────────────────────────
print(f"\n{SEP}")
print("[ 4 ] CROSS-VALIDATION Y EARLY STOPPING")
print(SEP)
print("Segun models/shots_predictor.py (lineas 41-45):")
print("  tscv = TimeSeriesSplit(n_splits=5)   <- CV temporal, correcto")
print("  cross_val_score(... scoring='r2')    <- metricas de CV")
print("  self.model.fit(X, y)                 <- fit final en TODO el dataset")
print()
print("  Cross-validation: SI (TimeSeriesSplit 5 folds)")
print("  Early stopping:   NO (fit sin eval_set ni early_stopping_rounds)")
print("  Problema: el modelo final entrena en todo X sin held-out set")
print("  -> Puede estar mas overfiteado que lo que indica el cv score")

# ── 5. PROBABILIDADES ALTAS EN LATAM ────────────────────────────
print(f"\n{SEP}")
print("[ 5 ] PROBABILIDADES 87-90% — ANALISIS DE SOBREAJUSTE")
print(SEP)

sot_dist = q("""
    SELECT m.league_name,
           ROUND(AVG(ps.shots_on_target)::numeric, 2) as avg_sot,
           ROUND(STDDEV(ps.shots_on_target)::numeric, 2) as std_sot,
           COUNT(*) as n,
           COUNT(CASE WHEN ps.shots_on_target >= 2 THEN 1 END) as ge2,
           ROUND(100.0 * COUNT(CASE WHEN ps.shots_on_target >= 2 THEN 1 END)
                 / COUNT(*), 1) as pct_ge2
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
    GROUP BY m.league_name
    ORDER BY avg_sot DESC
    LIMIT 15
""")
print(f"  {'Liga':<35} {'avg_SOT':>8} {'std':>6} {'n':>7} {'>=2 SOT%':>9}")
for r in sot_dist:
    print(f"  {str(r['league_name']):<35} {str(r['avg_sot']):>8} {str(r['std_sot']):>6} {r['n']:>7} {str(r['pct_ge2']):>8}%")

print("\nTop 10 shooters por avg_sot (>= 5 partidos):")
top_sh = q("""
    SELECT ps.player_name, m.league_name, COUNT(*) as partidos,
           ROUND(AVG(ps.shots_on_target)::numeric, 2) as avg_sot
    FROM player_stats ps
    JOIN matches m ON ps.match_id = m.id
    WHERE ps.shots_on_target IS NOT NULL AND ps.minutes_played > 45
    GROUP BY ps.player_name, m.league_name
    HAVING COUNT(*) >= 5
    ORDER BY avg_sot DESC
    LIMIT 10
""")
for r in top_sh:
    print(f"  {str(r['player_name']):<30} {str(r['league_name']):<25} {r['partidos']:>3}p  avg_sot={r['avg_sot']}")

print("\nJugadores predichos hoy con prob >85% vs su historial real:")
hoy = q("""
    SELECT p.prediction, p.probability, p.player_id
    FROM predictions p
    WHERE p.prediction_date = CURRENT_DATE
      AND p.market_type = 'player_shots'
      AND p.probability > 0.85
    ORDER BY p.probability DESC
""")
for r in hoy:
    print(f"\n  prob={r['probability']:.1%} | {r['prediction']}")
    if r['player_id']:
        hist = q1("""
            SELECT COUNT(*) as partidos,
                   ROUND(AVG(shots_on_target)::numeric, 2) as avg_sot,
                   COUNT(CASE WHEN shots_on_target >= 2 THEN 1 END) as ge2
            FROM player_stats
            WHERE player_id = :pid AND minutes_played > 45
        """, {"pid": r['player_id']})
        if hist['partidos']:
            real_pct = hist['ge2'] / hist['partidos'] * 100
            print(f"  Historial: {hist['partidos']}p | avg_SOT={hist['avg_sot']} | >= 2 SOT en {real_pct:.0f}% real (modelo dice {r['probability']:.0%})")
        else:
            print(f"  Sin historial en BD para player_id={r['player_id']}")

print(f"\n{SEP}")
print("FIN DEL ANALISIS")
print(f"{SEP}\n")
