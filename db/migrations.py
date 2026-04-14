"""
Migraciones: Crear todas las tablas del sistema en PostgreSQL (Supabase).
Ejecutar: python -m db.migrations
"""
from sqlalchemy import text
from db.connection import engine
from utils.logger import get_logger

logger = get_logger(__name__)

TABLES_SQL = """
-- Partidos historicos y del dia
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    api_fixture_id INTEGER UNIQUE NOT NULL,
    league_id INTEGER NOT NULL,
    league_name VARCHAR(100),
    season INTEGER,
    match_date TIMESTAMP NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(20),
    home_corners INTEGER,
    away_corners INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_possession FLOAT,
    away_possession FLOAT,
    venue VARCHAR(200),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Predicciones del modelo
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    prediction_date DATE NOT NULL,
    market_type VARCHAR(30) NOT NULL,
    prediction VARCHAR(100) NOT NULL,
    probability FLOAT NOT NULL,
    odds FLOAT,
    expected_value FLOAT,
    model_version VARCHAR(50),
    result VARCHAR(10),
    actual_outcome VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rendimiento diario
CREATE TABLE IF NOT EXISTS daily_performance (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy FLOAT,
    cumulative_total INTEGER DEFAULT 0,
    cumulative_correct INTEGER DEFAULT 0,
    cumulative_accuracy FLOAT,
    roi_simulated FLOAT,
    model_version VARCHAR(50),
    threshold_used FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Versiones del modelo
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(30),
    trained_at TIMESTAMP DEFAULT NOW(),
    training_samples INTEGER,
    accuracy_cv FLOAT,
    f1_score FLOAT,
    log_loss FLOAT,
    is_active BOOLEAN DEFAULT FALSE,
    model_binary BYTEA,
    feature_importance JSONB,
    notes TEXT
);

-- Stats de jugadores (para modelo de shots)
CREATE TABLE IF NOT EXISTS player_stats (
    id SERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(100),
    team_id INTEGER,
    team_name VARCHAR(100),
    league_id INTEGER,
    season INTEGER,
    match_id INTEGER REFERENCES matches(id),
    match_date DATE,
    shots_on_target INTEGER DEFAULT 0,
    shots_total INTEGER DEFAULT 0,
    minutes_played INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Historial de odds
CREATE TABLE IF NOT EXISTS odds_history (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    bookmaker VARCHAR(50),
    market VARCHAR(30),
    selection VARCHAR(50),
    odds_value FLOAT,
    captured_at TIMESTAMP DEFAULT NOW()
);

-- Sentimiento de noticias
CREATE TABLE IF NOT EXISTS news_sentiment (
    id SERIAL PRIMARY KEY,
    team_id INTEGER,
    team_name VARCHAR(100),
    headline TEXT,
    source VARCHAR(100),
    sentiment_score FLOAT,
    key_info TEXT,
    published_at TIMESTAMP,
    analyzed_at TIMESTAMP DEFAULT NOW()
);

-- Configuracion dinamica
CREATE TABLE IF NOT EXISTS config (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Valores iniciales de config (solo si no existen)
INSERT INTO config (key, value) VALUES
    ('current_threshold', '0.60'),
    ('min_threshold', '0.55'),
    ('max_threshold', '0.90'),
    ('max_daily_predictions', '4'),
    ('simulated_stake', '10000'),
    ('retrain_day', 'sunday'),
    ('model_active_1x2', 'v1.0'),
    ('model_active_corners', 'v1.0'),
    ('model_active_shots', 'v1.0')
ON CONFLICT (key) DO NOTHING;

-- Indices para mejorar rendimiento
CREATE INDEX IF NOT EXISTS idx_matches_fixture ON matches(api_fixture_id);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_history(match_id);
CREATE INDEX IF NOT EXISTS idx_news_team ON news_sentiment(team_id);
"""


def run_migrations():
    logger.info("Ejecutando migraciones...")
    try:
        with engine.connect() as conn:
            for statement in TABLES_SQL.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()
        logger.info("Migraciones ejecutadas exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error en migraciones: {e}")
        return False


if __name__ == "__main__":
    run_migrations()
