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

# Predicciones
MAX_DAILY_PREDICTIONS = 4
SIMULATED_STAKE = 10000  # COP

# Umbrales iniciales
INITIAL_THRESHOLD = 0.60
MIN_THRESHOLD = 0.55
MAX_THRESHOLD = 0.90
