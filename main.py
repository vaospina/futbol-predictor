"""
main.py - Entry point del sistema de predicción deportiva.

Worker puro para Render Background Worker.

Flujos programados (UTC):
  - 11:00 UTC (6:00 AM Colombia): Predicciones diarias
  - Cada 30 min 19:00–05:30 UTC (2:00 PM–12:30 AM Colombia): Evaluación
"""
from apscheduler.schedulers.blocking import BlockingScheduler

from flows.morning_predictions import run_daily_predictions
from flows.evening_results import run_evaluation_check
from db.migrations import run_migrations
from notifications.telegram import send_telegram
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== INICIANDO FUTBOL-PREDICTOR (worker) ===")

    run_migrations()

    scheduler = BlockingScheduler(timezone="UTC")

    # 6:00 AM Colombia = 11:00 UTC
    scheduler.add_job(
        run_daily_predictions,
        "cron",
        hour=11,
        minute=0,
        id="daily_predictions",
        replace_existing=True,
    )

    # Evaluación cada 30 min: 2PM–12:30AM Colombia = 19:00–05:30 UTC
    scheduler.add_job(
        run_evaluation_check,
        "cron",
        hour="19-23,0-5",
        minute="0,30",
        id="evaluation_check",
        replace_existing=True,
    )

    logger.info("Scheduler configurado con los siguientes jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.id} -> trigger: {job.trigger}")

    try:
        send_telegram("\U0001f7e2 *futbol-predictor* worker iniciado en Render")
    except Exception as e:
        logger.warning(f"No se pudo enviar Telegram de arranque: {e}")

    scheduler.start()


if __name__ == "__main__":
    main()
