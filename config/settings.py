import os
from dotenv import load_dotenv

load_dotenv()

# API-Football (conexion directa, NO RapidAPI)
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
API_FOOTBALL_BASE_URL = f"https://{API_FOOTBALL_HOST}"

# Base de datos (Supabase PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Config general
TIMEZONE = os.getenv("TIMEZONE", "America/Bogota")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Temporada actual
CURRENT_SEASON = 2025

# Modelo ML de corners desactivado: R² = -7.91% — predice peor que la media.
# Se reactivará cuando se entrene con métrica over/under (acc_cv > 55%).
ENABLE_CORNERS_MODEL = False

# Corners basado en reglas estadísticas puras (sin ML).
# Usa el promedio histórico de corners del equipo local en casa.
# Activado con 8.830 partidos disponibles y avg=9.62 corners/partido.
ENABLE_CORNERS_STATS = True

# Predicciones — sin límite diario; se muestran todas con prob>=umbral y EV>0
SIMULATED_STAKE = 10000  # COP

# Umbrales iniciales
INITIAL_THRESHOLD = 0.60
MIN_THRESHOLD = 0.55
MAX_THRESHOLD = 0.90
