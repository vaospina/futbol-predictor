"""
Post-backfill pipeline (autonomous):
  1. Wait for 2025 backfill log to show BACKFILL COMPLETADO
  2. Train all models (train_cached)
  3. Verify DB coverage (_check_bd)
  4. Send Telegram summary

Run: python -m scripts.pipeline_post_backfill
"""
import sys
import os
import time
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKFILL_LOG = BASE_DIR / "logs" / "backfill_2025_apr29.log"
PYTHON = sys.executable


# ── helpers ──────────────────────────────────────────────────────────────────

def wait_for_backfill(poll_sec=20, timeout_sec=7200):
    logger.info(f"Esperando que backfill 2025 termine (timeout {timeout_sec//60} min)...")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            content = BACKFILL_LOG.read_text(encoding="utf-8", errors="replace")
            if "BACKFILL COMPLETADO" in content:
                logger.info("Backfill 2025 COMPLETADO detectado en log.")
                return True
        except Exception as e:
            logger.warning(f"Error leyendo log: {e}")
        time.sleep(poll_sec)
    logger.error("Timeout esperando backfill — continuando de todas formas.")
    return False


def run_cmd(cmd, label, timeout=900):
    logger.info(f"=== {label} ===")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        combined = result.stdout + result.stderr
        logger.info(f"{label} salida (últimas 30 líneas):\n" +
                    "\n".join(combined.splitlines()[-30:]))
        return combined, result.returncode
    except subprocess.TimeoutExpired:
        logger.error(f"{label} timeout ({timeout}s)")
        return "", 1
    except Exception as e:
        logger.error(f"{label} error: {e}")
        return str(e), 1


def parse_train_metrics(output: str) -> dict:
    """Extract the JSON metrics block printed by train_cached."""
    # The script prints a JSON block to stdout at the end
    lines = output.splitlines()
    # Find the last '{' that starts a JSON block
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "{":
            try:
                block = "\n".join(lines[i:])
                return json.loads(block)
            except Exception:
                pass
    return {}


def build_telegram_message(metrics: dict, db_summary: str) -> str:
    from db.models import fetch_one, fetch_all

    total = fetch_one(
        "SELECT COUNT(*) as n, COUNT(CASE WHEN status='finished' THEN 1 END) as f FROM matches"
    )
    with_stats = fetch_one(
        "SELECT COUNT(*) as n FROM matches WHERE status='finished' AND home_corners IS NOT NULL"
    )
    leagues_2025 = fetch_all("""
        SELECT league_name,
               COUNT(CASE WHEN home_corners IS NOT NULL THEN 1 END) as con_stats
        FROM matches
        WHERE season = 2025 AND status = 'finished'
        GROUP BY league_name
        ORDER BY con_stats DESC
    """)

    # Read metrics from model_versions (authoritative source, never 0)
    def get_model(mtype):
        return fetch_one(
            "SELECT accuracy_cv, f1_score, log_loss, training_samples, notes "
            "FROM model_versions WHERE is_active=TRUE AND model_type=:t "
            "ORDER BY trained_at DESC LIMIT 1",
            {"t": mtype}
        ) or {}

    mv_1x2     = get_model("1x2")
    mv_shots   = get_model("player_shots")
    mv_corners = get_model("corners")

    acc    = mv_1x2.get("accuracy_cv") or 0
    ll     = mv_1x2.get("log_loss") or 0
    n_1x2  = mv_1x2.get("training_samples") or 0
    sr2    = mv_shots.get("accuracy_cv") or 0
    smae   = mv_shots.get("log_loss") or 0
    n_shots= mv_shots.get("training_samples") or 0
    cr2    = mv_corners.get("accuracy_cv") or 0
    cmae   = mv_corners.get("log_loss") or 0

    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    lines = [
        f"✅ *Backfill + Entrenamiento Completado*",
        f"_{now}_",
        "",
        "🗄 *Base de datos:*",
        f"  Total matches: {total['n']:,}",
        f"  Finished: {total['f']:,}",
        f"  Con estadísticas: {with_stats['n']:,}",
        "",
        "🤖 *Métricas del modelo:*",
        f"  1X2 — accuracy: {acc:.1%} | log-loss: {ll:.3f} | n={n_1x2:,}",
        f"  Shots — R²: {sr2:.3f} | MAE: {smae:.3f} | n={n_shots:,}",
        f"  Corners — R²: {cr2:.3f} | MAE: {cmae:.2f}",
        "",
        "📅 *Cobertura 2025 (ligas nuevas):*",
    ]
    for row in leagues_2025:
        icon = "✅" if row["con_stats"] > 0 else "❌"
        lines.append(f"  {icon} {row['league_name']}: {row['con_stats']} partidos")

    return "\n".join(lines)


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    logger.info("=== PIPELINE POST-BACKFILL INICIADO ===")

    # 1. Wait
    wait_for_backfill()

    # 2. Train
    train_out, train_rc = run_cmd(
        [PYTHON, "-m", "scripts.train_cached"],
        "ENTRENAMIENTO",
        timeout=900,
    )
    metrics = parse_train_metrics(train_out)
    if metrics:
        logger.info(f"Métricas extraídas: {json.dumps(metrics, indent=2)}")
    else:
        logger.warning("No se pudieron extraer métricas del entrenamiento.")

    # 3. Check BD
    check_out, check_rc = run_cmd(
        [PYTHON, str(BASE_DIR / "scripts" / "_check_bd.py")],
        "CHECK BD",
        timeout=60,
    )

    # 4. Telegram
    try:
        from notifications.telegram import send_telegram
        msg = build_telegram_message(metrics, check_out)
        send_telegram(msg)
        logger.info("Telegram enviado exitosamente.")
    except Exception as e:
        logger.error(f"Error enviando Telegram: {e}")
        # Fallback: send plain summary
        try:
            from notifications.telegram import send_telegram
            summary = (
                f"Pipeline completado {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
                f"Train RC={train_rc} | Check RC={check_rc}\n"
                f"Metricas: {json.dumps(metrics, default=str)[:500]}"
            )
            send_telegram(summary, parse_mode=None)
        except Exception:
            pass

    logger.info("=== PIPELINE COMPLETADO ===")


if __name__ == "__main__":
    main()
