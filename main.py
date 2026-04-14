"""
main.py - Entry point del sistema de predicción deportiva.

Worker puro para Render Background Worker: sin Flask, sin HTTP.
Sólo APScheduler bloqueante ejecutando los flujos programados.

Flujos programados (UTC):
  - 11:00 UTC (6:00 AM Colombia): Predicciones diarias
  - 03:00 UTC (10:00 PM Colombia): Evaluación de resultados
  - Domingos 07:00 UTC (2:00 AM Colombia): Re-entrenamiento semanal
"""
from apscheduler.schedulers.blocking import BlockingScheduler

from flows.morning_predictions import run_daily_predictions
from flows.evening_results import run_daily_results
from flows.weekly_retrain import run_weekly_retrain
from db.migrations import run_migrations
from notifications.telegram import send_telegram
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== INICIANDO FUTBOL-PREDICTOR (worker) ===")

    run_migrations()

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        run_daily_predictions,
        "cron",
        hour=11,
        minute=0,
        id="daily_predictions",
        replace_existing=True,
    )
    scheduler.add_job(
        run_daily_results,
        "cron",
        hour=3,
        minute=0,
        id="daily_results",
        replace_existing=True,
    )
    scheduler.add_job(
        run_weekly_retrain,
        "cron",
        day_of_week="sun",
        hour=7,
        minute=0,
        id="weekly_retrain",
        replace_existing=True,
    )

    logger.info("Scheduler iniciado con los siguientes jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.id} -> next run: {job.next_run_time}")

    try:
        send_telegram("\U0001f7e2 *futbol-predictor* worker iniciado en Render")
    except Exception as e:
        logger.warning(f"No se pudo enviar Telegram de arranque: {e}")

    scheduler.start()


if __name__ == "__main__":
    main()
