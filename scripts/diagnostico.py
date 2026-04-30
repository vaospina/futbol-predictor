"""
Diagnóstico completo del sistema de predicciones.
Ejecutar desde la raíz del proyecto: python scripts/diagnostico.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datetime import date, timedelta
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL no encontrada en .env")
    sys.exit(1)

engine = create_engine(DATABASE_URL)

def q(sql, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols = result.keys()
        return [dict(zip(cols, row)) for row in result.fetchall()]

def q1(sql, params=None):
    rows = q(sql, params)
    return rows[0] if rows else {}

SEP = "=" * 60

print(f"\n{SEP}")
print("  DIAGNÓSTICO FUTBOL-PREDICTOR")
print(f"  Fecha actual: {date.today()}")
print(f"{SEP}\n")

# ── 1. ESTADO DE TABLAS ──────────────────────────────────────────
print("[ 1 ] CONTEO DE REGISTROS POR TABLA")
for tabla in ["matches", "predictions", "daily_performance", "model_versions", "config"]:
    try:
        r = q1(f"SELECT COUNT(*) as n FROM {tabla}")
        print(f"  {tabla:25s}: {r.get('n', '?'):>7} registros")
    except Exception as e:
        print(f"  {tabla:25s}: ERROR - {e}")

# ── 2. ÚLTIMAS PREDICCIONES ──────────────────────────────────────
print(f"\n[ 2 ] ÚLTIMAS 20 PREDICCIONES")
preds = q("""
    SELECT p.id, p.prediction_date, p.market_type, p.prediction,
           ROUND(p.probability::numeric, 3) as prob,
           ROUND(p.odds::numeric, 2) as odds,
           ROUND(p.expected_value::numeric, 3) as ev,
           p.result, p.model_version, p.data_source,
           m.home_team, m.away_team, m.league_name
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    ORDER BY p.created_at DESC LIMIT 20
""")
if not preds:
    print("  !! SIN PREDICCIONES EN LA BASE DE DATOS !!")
else:
    last = preds[0]
    print(f"  Última predicción: {last['prediction_date']} | {last['home_team']} vs {last['away_team']}")
    print(f"  Días sin predecir desde la última: {(date.today() - last['prediction_date']).days}")
    for p in preds:
        result_str = p['result'] or 'pendiente'
        print(f"  {p['prediction_date']} | {p['league_name'][:20]:20s} | "
              f"{p['market_type']:12s} | prob={p['prob']} ev={p['ev']} | {result_str}")

# ── 3. TIMELINE ÚLTIMOS 30 DÍAS ──────────────────────────────────
print(f"\n[ 3 ] TIMELINE PREDICCIONES (últimos 30 días)")
since = date.today() - timedelta(days=30)
timeline = q("""
    SELECT prediction_date,
           COUNT(*) as total,
           COUNT(CASE WHEN result='win' THEN 1 END) as wins,
           COUNT(CASE WHEN result='loss' THEN 1 END) as losses,
           COUNT(CASE WHEN result IS NULL THEN 1 END) as pendientes
    FROM predictions
    WHERE prediction_date >= :since
    GROUP BY prediction_date
    ORDER BY prediction_date DESC
""", {"since": since})
if not timeline:
    print("  Sin predicciones en los últimos 30 días")
else:
    total_preds = sum(r['total'] for r in timeline)
    dias_con_preds = len(timeline)
    print(f"  Días con predicciones: {dias_con_preds} / 30")
    print(f"  Total predicciones: {total_preds}")
    print(f"  {'Fecha':12s} {'Total':>6} {'Win':>5} {'Loss':>6} {'Pend':>6}")
    for r in timeline:
        print(f"  {str(r['prediction_date']):12s} {r['total']:>6} {r['wins']:>5} {r['losses']:>6} {r['pendientes']:>6}")

# ── 4. DÍAS SIN PREDICCIONES ─────────────────────────────────────
print(f"\n[ 4 ] DÍAS SIN PREDICCIONES (últimos 30 días)")
dias_con = {r['prediction_date'] for r in timeline} if timeline else set()
dias_sin = []
for i in range(30):
    d = date.today() - timedelta(days=i)
    if d not in dias_con:
        dias_sin.append(d)
print(f"  {len(dias_sin)} días sin predicciones de los últimos 30")
if dias_sin[:10]:
    print(f"  Más recientes: {[str(d) for d in dias_sin[:10]]}")

# ── 5. DAILY PERFORMANCE ─────────────────────────────────────────
print(f"\n[ 5 ] DAILY PERFORMANCE (últimos 14 días)")
perf = q("""
    SELECT date, total_predictions, correct_predictions,
           ROUND(accuracy::numeric, 3) as accuracy,
           ROUND(roi_simulated::numeric, 2) as roi,
           threshold_used, model_version
    FROM daily_performance
    ORDER BY date DESC LIMIT 14
""")
if not perf:
    print("  !! SIN REGISTROS EN daily_performance !!")
else:
    for p in perf:
        print(f"  {p['date']} | total={p['total_predictions']} "
              f"correct={p['correct_predictions']} acc={p['accuracy']} "
              f"roi={p['roi']} threshold={p['threshold_used']}")

# ── 6. ACCURACY POR LIGA ──────────────────────────────────────────
print(f"\n[ 6 ] ACCURACY POR LIGA (últimos 7 días)")
acc_liga = q("""
    SELECT m.league_name,
           COUNT(*) as total,
           COUNT(CASE WHEN p.result='win' THEN 1 END) as wins,
           ROUND(100.0 * COUNT(CASE WHEN p.result='win' THEN 1 END) / NULLIF(COUNT(CASE WHEN p.result IS NOT NULL THEN 1 END), 0), 1) as accuracy_pct
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    WHERE p.prediction_date >= :since
    GROUP BY m.league_name
    ORDER BY total DESC
""", {"since": date.today() - timedelta(days=7)})
if not acc_liga:
    print("  Sin datos de los últimos 7 días")
else:
    for r in acc_liga:
        print(f"  {r['league_name'][:30]:30s} | total={r['total']} wins={r['wins']} acc={r['accuracy_pct']}%")

# ── 7. MODEL VERSIONS ─────────────────────────────────────────────
print(f"\n[ 7 ] MODEL VERSIONS")
mvs = q("""
    SELECT version, model_type, is_active,
           ROUND(accuracy_cv::numeric, 4) as acc_cv,
           training_samples, trained_at
    FROM model_versions
    ORDER BY trained_at DESC LIMIT 10
""")
if not mvs:
    print("  !! SIN MODELOS REGISTRADOS !!")
else:
    for mv in mvs:
        active_str = " [ACTIVO]" if mv['is_active'] else ""
        print(f"  {mv['trained_at']} | {mv['model_type']:15s} {mv['version']:10s} "
              f"acc_cv={mv['acc_cv']} samples={mv['training_samples']}{active_str}")

# ── 8. CONFIG TABLE ──────────────────────────────────────────────
print(f"\n[ 8 ] CONFIG")
configs = q("SELECT key, value, updated_at FROM config ORDER BY key")
if not configs:
    print("  !! TABLA CONFIG VACÍA - Sin umbral dinámico configurado !!")
else:
    for c in configs:
        print(f"  {c['key']:30s} = {c['value']}  (updated: {c['updated_at']})")

# ── 9. MATCHES RECIENTES Y PENDIENTES ────────────────────────────
print(f"\n[ 9 ] MATCHES (últimos 7 días + próximos 3 días)")
matches = q("""
    SELECT DATE(match_date) as dia, status,
           COUNT(*) as n,
           COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as con_resultado
    FROM matches
    WHERE match_date >= NOW() - INTERVAL '7 days'
      AND match_date <= NOW() + INTERVAL '3 days'
    GROUP BY dia, status
    ORDER BY dia DESC, status
""")
if not matches:
    print("  Sin partidos recientes en BD")
else:
    for r in matches:
        print(f"  {r['dia']} | status={r['status']:12s} | total={r['n']} con_resultado={r['con_resultado']}")

# ── 10. ANÁLISIS DE FILTROS (por qué no pasan el umbral) ──────────
print(f"\n[ 10 ] ANÁLISIS: ¿POR QUÉ SE DESCARTAN PREDICCIONES?")
# Verificar el umbral actual
current_threshold = q1("SELECT value FROM config WHERE key='current_threshold'")
threshold = float(current_threshold.get('value', 0.60)) if current_threshold else 0.60
print(f"  Umbral actual: {threshold:.0%}")

# Ver distribución de probabilidades en predicciones recientes
prob_dist = q("""
    SELECT
        COUNT(*) as total_preds,
        ROUND(AVG(probability::numeric), 3) as avg_prob,
        ROUND(MIN(probability::numeric), 3) as min_prob,
        ROUND(MAX(probability::numeric), 3) as max_prob,
        COUNT(CASE WHEN expected_value > 0 THEN 1 END) as ev_positivo
    FROM predictions
    WHERE prediction_date >= :since
""", {"since": date.today() - timedelta(days=30)})
if prob_dist and prob_dist[0]['total_preds']:
    pd = prob_dist[0]
    print(f"  En últimos 30 días: avg_prob={pd['avg_prob']} "
          f"min={pd['min_prob']} max={pd['max_prob']} EV+={pd['ev_positivo']}/{pd['total_preds']}")

# ── 11. HIPÓTESIS DIAGNÓSTICO ─────────────────────────────────────
print(f"\n{SEP}")
print("  DIAGNÓSTICO - HIPÓTESIS")
print(f"{SEP}")

total_preds_total = q1("SELECT COUNT(*) as n FROM predictions")['n']
active_model = q1("SELECT COUNT(*) as n FROM model_versions WHERE is_active=TRUE")['n']
config_count = q1("SELECT COUNT(*) as n FROM config")['n']
preds_30d = len(timeline) if timeline else 0

if total_preds_total == 0:
    print("\n  CRITICO: No hay ninguna predicción en la BD.")
    print("  → El flujo nunca se ha ejecutado exitosamente,")
    print("    o los modelos nunca fueron entrenados/guardados.")

if active_model == 0:
    print("\n  CRITICO: No hay modelos activos en model_versions.")
    print("  → Ejecutar el entrenamiento: python scripts/train_cached.py")

if config_count == 0:
    print("\n  AVISO: Tabla config vacía (sin umbral dinámico).")
    print("  → get_current_threshold() devuelve 0.60 por defecto (OK).")

if preds_30d < 10:
    print(f"\n  AVISO: Solo {preds_30d} días con predicciones en 30 días.")
    print("  → El scheduler en Render puede no estar corriendo correctamente.")
    print("  → Verificar logs de Render para el worker.")

if threshold >= 0.75:
    print(f"\n  AVISO: Umbral muy alto ({threshold:.0%}).")
    print("  → Pocas predicciones pasarán. Considerar bajar a 0.60.")

print("\n  PASOS SIGUIENTES SUGERIDOS:")
print("  1. Verificar que el worker de Render está activo y los logs muestran")
print("     '=== INICIANDO FLUJO DE PREDICCIONES ===' cada mañana")
print("  2. Revisar si los modelos .pkl existen localmente (models/*.pkl)")
print("  3. Ejecutar manualmente: python -c \"from flows.morning_predictions import run_daily_predictions; run_daily_predictions()\"")
print("  4. Verificar cuota API-Football (100 req/día en plan gratuito)")
print(f"{SEP}\n")
