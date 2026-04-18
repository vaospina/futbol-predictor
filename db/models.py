"""
Funciones de acceso a datos para todas las tablas.
"""
from datetime import date, datetime
from sqlalchemy import text
from db.connection import engine
from utils.logger import get_logger

logger = get_logger(__name__)


def execute_query(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        conn.commit()
        return result


def fetch_all(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


def fetch_one(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        row = result.fetchone()
        data = dict(zip(result.keys(), row)) if row else None
        conn.commit()
        return data


# === MATCHES ===

def upsert_match(match_data: dict):
    query = """
    INSERT INTO matches (
        api_fixture_id, league_id, league_name, season, match_date,
        home_team, away_team, home_team_id, away_team_id,
        home_score, away_score, status,
        home_corners, away_corners,
        home_shots_on_target, away_shots_on_target,
        home_possession, away_possession, venue
    ) VALUES (
        :api_fixture_id, :league_id, :league_name, :season, :match_date,
        :home_team, :away_team, :home_team_id, :away_team_id,
        :home_score, :away_score, :status,
        :home_corners, :away_corners,
        :home_shots_on_target, :away_shots_on_target,
        :home_possession, :away_possession, :venue
    )
    ON CONFLICT (api_fixture_id) DO UPDATE SET
        home_score = EXCLUDED.home_score,
        away_score = EXCLUDED.away_score,
        status = EXCLUDED.status,
        home_corners = EXCLUDED.home_corners,
        away_corners = EXCLUDED.away_corners,
        home_shots_on_target = EXCLUDED.home_shots_on_target,
        away_shots_on_target = EXCLUDED.away_shots_on_target,
        home_possession = EXCLUDED.home_possession,
        away_possession = EXCLUDED.away_possession
    RETURNING id
    """
    result = fetch_one(query, match_data)
    return result["id"] if result else None


def get_match_by_fixture_id(fixture_id: int):
    return fetch_one(
        "SELECT * FROM matches WHERE api_fixture_id = :fid",
        {"fid": fixture_id}
    )


def get_matches_by_date(match_date: date):
    return fetch_all(
        "SELECT * FROM matches WHERE DATE(match_date) = :d ORDER BY match_date",
        {"d": match_date}
    )


def get_team_last_matches(team_id: int, n: int = 10, before_date: date = None):
    before = before_date or date.today()
    return fetch_all(
        """SELECT * FROM matches
        WHERE (home_team_id = :tid OR away_team_id = :tid)
          AND status = 'finished' AND DATE(match_date) < :bd
        ORDER BY match_date DESC LIMIT :n""",
        {"tid": team_id, "bd": before, "n": n}
    )


def get_team_home_matches(team_id: int, n: int = 10, before_date: date = None):
    before = before_date or date.today()
    return fetch_all(
        """SELECT * FROM matches
        WHERE home_team_id = :tid AND status = 'finished' AND DATE(match_date) < :bd
        ORDER BY match_date DESC LIMIT :n""",
        {"tid": team_id, "bd": before, "n": n}
    )


def get_team_away_matches(team_id: int, n: int = 10, before_date: date = None):
    before = before_date or date.today()
    return fetch_all(
        """SELECT * FROM matches
        WHERE away_team_id = :tid AND status = 'finished' AND DATE(match_date) < :bd
        ORDER BY match_date DESC LIMIT :n""",
        {"tid": team_id, "bd": before, "n": n}
    )


def get_h2h_matches(team1_id: int, team2_id: int, n: int = 5, before_date: date = None):
    before = before_date or date.today()
    return fetch_all(
        """SELECT * FROM matches
        WHERE ((home_team_id = :t1 AND away_team_id = :t2)
            OR (home_team_id = :t2 AND away_team_id = :t1))
          AND status = 'finished' AND DATE(match_date) < :bd
        ORDER BY match_date DESC LIMIT :n""",
        {"t1": team1_id, "t2": team2_id, "n": n, "bd": before}
    )


# === PREDICTIONS ===

def insert_prediction(pred: dict):
    query = """
    INSERT INTO predictions (
        match_id, prediction_date, market_type, prediction,
        probability, odds, expected_value, model_version, data_source
    ) VALUES (
        :match_id, :prediction_date, :market_type, :prediction,
        :probability, :odds, :expected_value, :model_version, :data_source
    ) RETURNING id
    """
    if "data_source" not in pred:
        pred["data_source"] = "api_real"
    result = fetch_one(query, pred)
    return result["id"] if result else None


def get_predictions_by_date(pred_date: date):
    return fetch_all(
        """SELECT p.*, m.home_team, m.away_team, m.league_name,
                  m.home_score, m.away_score, m.home_corners, m.away_corners
        FROM predictions p
        JOIN matches m ON p.match_id = m.id
        WHERE p.prediction_date = :d
        ORDER BY p.probability DESC""",
        {"d": pred_date}
    )


def get_pending_predictions(pred_date: date):
    return fetch_all(
        """SELECT p.*, m.home_team, m.away_team, m.api_fixture_id,
                  m.home_score, m.away_score, m.home_corners, m.away_corners,
                  m.home_shots_on_target, m.away_shots_on_target, m.status,
                  m.league_id, m.league_name, m.season,
                  m.home_team_id, m.away_team_id
        FROM predictions p
        JOIN matches m ON p.match_id = m.id
        WHERE p.prediction_date = :d AND p.result IS NULL""",
        {"d": pred_date}
    )


def get_recent_pending_predictions(lookback_days: int = 2):
    from datetime import timedelta
    from_date = date.today() - timedelta(days=lookback_days)
    return fetch_all(
        """SELECT p.*, m.home_team, m.away_team, m.api_fixture_id,
                  m.home_score, m.away_score, m.home_corners, m.away_corners,
                  m.home_shots_on_target, m.away_shots_on_target, m.status,
                  m.league_id, m.league_name, m.season,
                  m.home_team_id, m.away_team_id
        FROM predictions p
        JOIN matches m ON p.match_id = m.id
        WHERE p.prediction_date >= :from_date AND p.result IS NULL
        ORDER BY p.prediction_date, p.id""",
        {"from_date": from_date}
    )


def update_prediction_result(pred_id: int, result: str, actual_outcome: str):
    execute_query(
        """UPDATE predictions SET result = :result, actual_outcome = :outcome
        WHERE id = :pid""",
        {"result": result, "outcome": actual_outcome, "pid": pred_id}
    )


# === DAILY PERFORMANCE ===

def upsert_daily_performance(perf: dict):
    query = """
    INSERT INTO daily_performance (
        date, total_predictions, correct_predictions, accuracy,
        cumulative_total, cumulative_correct, cumulative_accuracy,
        roi_simulated, model_version, threshold_used
    ) VALUES (
        :date, :total_predictions, :correct_predictions, :accuracy,
        :cumulative_total, :cumulative_correct, :cumulative_accuracy,
        :roi_simulated, :model_version, :threshold_used
    )
    ON CONFLICT (date) DO UPDATE SET
        total_predictions = EXCLUDED.total_predictions,
        correct_predictions = EXCLUDED.correct_predictions,
        accuracy = EXCLUDED.accuracy,
        cumulative_total = EXCLUDED.cumulative_total,
        cumulative_correct = EXCLUDED.cumulative_correct,
        cumulative_accuracy = EXCLUDED.cumulative_accuracy,
        roi_simulated = EXCLUDED.roi_simulated
    """
    execute_query(query, perf)


def get_cumulative_stats():
    return fetch_one(
        """SELECT COALESCE(SUM(total_predictions), 0) as total,
                  COALESCE(SUM(correct_predictions), 0) as correct
        FROM daily_performance"""
    )


def get_current_streak():
    """Calcula la racha actual (positiva = aciertos, negativa = fallos)."""
    preds = fetch_all(
        """SELECT result FROM predictions
        WHERE result IS NOT NULL
        ORDER BY created_at DESC LIMIT 50"""
    )
    if not preds:
        return 0
    streak = 0
    first_result = preds[0]["result"]
    for p in preds:
        if p["result"] == first_result:
            streak += 1
        else:
            break
    return streak if first_result == "win" else -streak


def get_worst_streak():
    """Peor racha de fallos consecutivos."""
    preds = fetch_all(
        """SELECT result FROM predictions
        WHERE result IS NOT NULL
        ORDER BY created_at ASC"""
    )
    max_losses = 0
    current_losses = 0
    for p in preds:
        if p["result"] == "loss":
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    return max_losses


def get_accuracy_by_type():
    return fetch_all(
        """SELECT market_type,
                  COUNT(*) as total,
                  SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as correct,
                  ROUND(AVG(CASE WHEN result = 'win' THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy
        FROM predictions WHERE result IS NOT NULL
        GROUP BY market_type"""
    )


# === MODEL VERSIONS ===

def insert_model_version(model_data: dict):
    query = """
    INSERT INTO model_versions (
        version, model_type, training_samples, accuracy_cv,
        f1_score, log_loss, is_active, model_binary,
        feature_importance, notes
    ) VALUES (
        :version, :model_type, :training_samples, :accuracy_cv,
        :f1_score, :log_loss, :is_active, :model_binary,
        :feature_importance, :notes
    ) RETURNING id
    """
    result = fetch_one(query, model_data)
    return result["id"] if result else None


def get_active_model(model_type: str):
    return fetch_one(
        "SELECT * FROM model_versions WHERE model_type = :mt AND is_active = TRUE",
        {"mt": model_type}
    )


def deactivate_models(model_type: str):
    execute_query(
        "UPDATE model_versions SET is_active = FALSE WHERE model_type = :mt",
        {"mt": model_type}
    )


def activate_model(version: str):
    execute_query(
        "UPDATE model_versions SET is_active = TRUE WHERE version = :v",
        {"v": version}
    )


# === PLAYER STATS ===

def upsert_player_stat(stat: dict):
    query = """
    INSERT INTO player_stats (
        player_id, player_name, team_id, team_name,
        league_id, season, match_id, match_date,
        shots_on_target, shots_total, minutes_played
    ) VALUES (
        :player_id, :player_name, :team_id, :team_name,
        :league_id, :season, :match_id, :match_date,
        :shots_on_target, :shots_total, :minutes_played
    )
    ON CONFLICT DO NOTHING
    """
    execute_query(query, stat)


def batch_upsert_player_stats(stats: list):
    if not stats:
        return
    query = """
    INSERT INTO player_stats (
        player_id, player_name, team_id, team_name,
        league_id, season, match_id, match_date,
        shots_on_target, shots_total, minutes_played
    ) VALUES (
        :player_id, :player_name, :team_id, :team_name,
        :league_id, :season, :match_id, :match_date,
        :shots_on_target, :shots_total, :minutes_played
    )
    ON CONFLICT DO NOTHING
    """
    with engine.connect() as conn:
        conn.execute(text(query), stats)
        conn.commit()


def get_player_recent_stats(player_id: int, n: int = 10, before_date: date = None):
    before = before_date or date.today()
    return fetch_all(
        """SELECT * FROM player_stats
        WHERE player_id = :pid AND match_date < :bd
        ORDER BY match_date DESC LIMIT :n""",
        {"pid": player_id, "n": n, "bd": before}
    )


def get_top_shooters_by_league(league_id: int, season: int, min_matches: int = 5):
    return fetch_all(
        """SELECT player_id, player_name, team_name,
                  COUNT(*) as matches,
                  AVG(shots_on_target) as avg_sot,
                  AVG(shots_total) as avg_shots
        FROM player_stats
        WHERE league_id = :lid AND season = :s
        GROUP BY player_id, player_name, team_name
        HAVING COUNT(*) >= :mm
        ORDER BY avg_sot DESC LIMIT 20""",
        {"lid": league_id, "s": season, "mm": min_matches}
    )


# === ODDS HISTORY ===

def insert_odds(odds_data: dict):
    query = """
    INSERT INTO odds_history (match_id, bookmaker, market, selection, odds_value)
    VALUES (:match_id, :bookmaker, :market, :selection, :odds_value)
    """
    execute_query(query, odds_data)


def get_odds_for_match(match_id: int):
    return fetch_all(
        "SELECT * FROM odds_history WHERE match_id = :mid",
        {"mid": match_id}
    )


# === NEWS SENTIMENT ===

def insert_sentiment(sentiment: dict):
    query = """
    INSERT INTO news_sentiment (
        team_id, team_name, headline, source,
        sentiment_score, key_info, published_at
    ) VALUES (
        :team_id, :team_name, :headline, :source,
        :sentiment_score, :key_info, :published_at
    )
    """
    execute_query(query, sentiment)


def get_team_sentiment(team_id: int, days: int = 3):
    return fetch_all(
        """SELECT * FROM news_sentiment
        WHERE team_id = :tid AND analyzed_at > NOW() - INTERVAL ':days days'
        ORDER BY analyzed_at DESC""",
        {"tid": team_id, "days": days}
    )


def get_team_avg_sentiment(team_id: int, days: int = 3):
    result = fetch_one(
        """SELECT AVG(sentiment_score) as avg_score,
                  BOOL_OR(key_info LIKE '%%lesion%%' OR key_info LIKE '%%injury%%'
                          OR key_info LIKE '%%suspension%%') as key_player_absent
        FROM news_sentiment
        WHERE team_id = :tid AND analyzed_at > NOW() - INTERVAL '3 days'""",
        {"tid": team_id}
    )
    return result


# === CONFIG ===

def get_config(key: str):
    result = fetch_one("SELECT value FROM config WHERE key = :k", {"k": key})
    return result["value"] if result else None


def set_config(key: str, value: str):
    execute_query(
        """INSERT INTO config (key, value, updated_at)
        VALUES (:k, :v, NOW())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
        {"k": key, "v": value}
    )
