"""
Construccion de features para los modelos ML.
Todas las features definidas en la seccion 5.1 del documento.
"""
import numpy as np
from datetime import date, timedelta
from db.models import (
    get_team_home_matches, get_team_away_matches, get_team_last_matches,
    get_h2h_matches, get_player_recent_stats,
)
from data.odds_collector import get_match_odds_summary
from config.leagues import DERBIES
from utils.helpers import safe_float, safe_int
from utils.logger import get_logger

logger = get_logger(__name__)


def build_match_features(match: dict, sentiment_home: dict = None, sentiment_away: dict = None) -> dict:
    """Construye el vector completo de features para un partido."""
    home_id = match.get("home_team_id")
    away_id = match.get("away_team_id")
    match_date = match.get("match_date")

    if isinstance(match_date, str):
        match_date = date.fromisoformat(match_date[:10])
    elif hasattr(match_date, "date"):
        match_date = match_date.date()

    # === Rendimiento del equipo (ultimos 10 partidos) ===
    home_matches_h = get_team_home_matches(home_id, 10, match_date)
    away_matches_a = get_team_away_matches(away_id, 10, match_date)
    home_all = get_team_last_matches(home_id, 10, match_date)
    away_all = get_team_last_matches(away_id, 10, match_date)

    features = {}

    # Win rates
    features["home_win_rate_last10"] = _calc_win_rate(home_matches_h, home_id, is_home=True)
    features["away_win_rate_last10"] = _calc_win_rate(away_matches_a, away_id, is_home=False)

    # Goals averages
    features["home_goals_scored_avg"] = _avg_goals_scored(home_matches_h, home_id, is_home=True)
    features["away_goals_scored_avg"] = _avg_goals_scored(away_matches_a, away_id, is_home=False)
    features["home_goals_conceded_avg"] = _avg_goals_conceded(home_matches_h, home_id, is_home=True)
    features["away_goals_conceded_avg"] = _avg_goals_conceded(away_matches_a, away_id, is_home=False)

    # Corners averages
    features["home_corners_avg"] = _avg_stat(home_matches_h, "home_corners")
    features["away_corners_avg"] = _avg_stat(away_matches_a, "away_corners")

    # Shots on target averages
    features["home_shots_on_target_avg"] = _avg_stat(home_matches_h, "home_shots_on_target")
    features["away_shots_on_target_avg"] = _avg_stat(away_matches_a, "away_shots_on_target")

    # === Head to head (solo partidos anteriores a match_date) ===
    h2h = get_h2h_matches(home_id, away_id, 5, before_date=match_date)
    features["h2h_home_wins"] = sum(
        1 for m in h2h
        if (m["home_team_id"] == home_id and (m.get("home_score") or 0) > (m.get("away_score") or 0))
        or (m["away_team_id"] == home_id and (m.get("away_score") or 0) > (m.get("home_score") or 0))
    )
    features["h2h_draws"] = sum(
        1 for m in h2h if m.get("home_score") == m.get("away_score") and m.get("home_score") is not None
    )
    features["h2h_avg_corners"] = np.mean([
        safe_int(m.get("home_corners")) + safe_int(m.get("away_corners"))
        for m in h2h if m.get("home_corners") is not None
    ]) if h2h else 0.0

    # === Forma reciente (ultimos 5) ===
    home_last5 = get_team_last_matches(home_id, 5, match_date)
    away_last5 = get_team_last_matches(away_id, 5, match_date)
    features["home_form_points"] = _calc_form_points(home_last5, home_id)
    features["away_form_points"] = _calc_form_points(away_last5, away_id)
    features["home_form_streak"] = _calc_streak(home_last5, home_id)
    features["away_form_streak"] = _calc_streak(away_last5, away_id)

    # === Features específicas para goles ===
    features["home_goals_avg_last5"] = _avg_goals_total(home_last5)
    features["away_goals_avg_last5"] = _avg_goals_total(away_last5)
    features["home_over25_rate"] = _over_rate(home_last5, 2.5)
    features["away_over25_rate"] = _over_rate(away_last5, 2.5)
    features["h2h_avg_goals"] = _avg_goals_total(h2h)
    features["home_ht_goals_avg"] = _avg_ht_goals(home_last5, home_id, is_home=True)
    features["away_ht_goals_avg"] = _avg_ht_goals(away_last5, away_id, is_home=False)

    # === Posicion en la tabla ===
    features["home_league_position"] = match.get("home_league_position", 10)
    features["away_league_position"] = match.get("away_league_position", 10)
    features["position_difference"] = features["away_league_position"] - features["home_league_position"]

    # === Odds del mercado ===
    match_id = match.get("id") or match.get("match_id")
    odds = get_match_odds_summary(match_id) if match_id else {}
    features["odds_home_win"] = safe_float(odds.get("odds_home_win"), 2.0)
    features["odds_draw"] = safe_float(odds.get("odds_draw"), 3.3)
    features["odds_away_win"] = safe_float(odds.get("odds_away_win"), 3.5)
    features["odds_over_corners"] = safe_float(odds.get("odds_over_corners"), 1.9)
    # Prob implicita normalizada
    total_prob = (1 / features["odds_home_win"] + 1 / features["odds_draw"] + 1 / features["odds_away_win"])
    features["odds_implied_prob_home"] = (1 / features["odds_home_win"]) / total_prob if total_prob > 0 else 0.33

    # === Contexto ===
    home_team = match.get("home_team", "")
    away_team = match.get("away_team", "")
    features["is_derby"] = 1 if (home_team, away_team) in DERBIES else 0

    features["days_since_last_match_home"] = _days_since_last(home_all, match_date)
    features["days_since_last_match_away"] = _days_since_last(away_all, match_date)

    # === Sentimiento de noticias ===
    sent_h = sentiment_home or {}
    sent_a = sentiment_away or {}
    features["home_sentiment_score"] = safe_float(sent_h.get("sentiment_score"), 0.0)
    features["away_sentiment_score"] = safe_float(sent_a.get("sentiment_score"), 0.0)
    features["home_key_player_absent"] = 1 if sent_h.get("key_player_absent") else 0
    features["away_key_player_absent"] = 1 if sent_a.get("key_player_absent") else 0

    return features


def build_corners_features(match_features: dict) -> dict:
    """Features especificas para el modelo de corners."""
    keys = [
        "home_corners_avg", "away_corners_avg", "h2h_avg_corners",
        "home_shots_on_target_avg", "away_shots_on_target_avg",
        "home_form_points", "away_form_points",
        "position_difference", "is_derby",
        "odds_over_corners",
        "home_league_position", "away_league_position",
    ]
    return {k: match_features.get(k, 0) for k in keys}


def build_player_features(player_id: int, match_features: dict, before_date=None) -> dict:
    """Features para el modelo de shots on target de un jugador.

    before_date: si se provee, sólo usa estadísticas de partidos estrictamente
    anteriores. Esencial para evitar data leakage durante entrenamiento.
    """
    recent = get_player_recent_stats(player_id, 10, before_date=before_date)

    if not recent:
        return None

    sot_vals = [s.get("shots_on_target", 0) for s in recent]
    shots_vals = [s.get("shots_total", 0) for s in recent]
    mins_vals = [s.get("minutes_played", 0) for s in recent]
    goals_vals = [s.get("goals", 0) or 0 for s in recent]

    last3 = recent[:3]
    sot_last3 = [s.get("shots_on_target", 0) for s in last3] if len(last3) >= 3 else sot_vals
    mins_last3 = [s.get("minutes_played", 0) for s in last3] if len(last3) >= 3 else mins_vals
    goals_last3 = [s.get("goals", 0) or 0 for s in last3] if len(last3) >= 3 else goals_vals

    return {
        "player_avg_sot": np.mean(sot_vals),
        "player_avg_shots_total": np.mean(shots_vals),
        "player_avg_minutes": np.mean(mins_vals),
        "player_matches_count": len(recent),
        "player_sot_last3": np.mean(sot_last3),
        "player_minutes_last3": np.mean(mins_last3),
        "opponent_goals_conceded_avg": match_features.get("away_goals_conceded_avg", 0),
        "team_shots_on_target_avg": match_features.get("home_shots_on_target_avg", 0),
        "is_derby": match_features.get("is_derby", 0),
        "is_home": match_features.get("is_home", 1),
        "player_goals_last3": np.sum(goals_last3),
    }


# === Helpers ===

def _calc_win_rate(matches: list, team_id: int, is_home: bool) -> float:
    if not matches:
        return 0.0
    wins = 0
    for m in matches:
        hs = m.get("home_score")
        aws = m.get("away_score")
        if hs is None or aws is None:
            continue
        if is_home and hs > aws:
            wins += 1
        elif not is_home and aws > hs:
            wins += 1
    return wins / len(matches)


def _avg_goals_scored(matches: list, team_id: int, is_home: bool) -> float:
    if not matches:
        return 0.0
    key = "home_score" if is_home else "away_score"
    goals = [safe_int(m.get(key)) for m in matches if m.get(key) is not None]
    return np.mean(goals) if goals else 0.0


def _avg_goals_conceded(matches: list, team_id: int, is_home: bool) -> float:
    if not matches:
        return 0.0
    key = "away_score" if is_home else "home_score"
    goals = [safe_int(m.get(key)) for m in matches if m.get(key) is not None]
    return np.mean(goals) if goals else 0.0


def _avg_stat(matches: list, stat_key: str) -> float:
    if not matches:
        return 0.0
    vals = [safe_float(m.get(stat_key)) for m in matches if m.get(stat_key) is not None]
    return np.mean(vals) if vals else 0.0


def _calc_form_points(matches: list, team_id: int) -> int:
    points = 0
    for m in matches:
        hs = m.get("home_score")
        aws = m.get("away_score")
        if hs is None or aws is None:
            continue
        is_home = m.get("home_team_id") == team_id
        if is_home:
            if hs > aws:
                points += 3
            elif hs == aws:
                points += 1
        else:
            if aws > hs:
                points += 3
            elif hs == aws:
                points += 1
    return points


def _calc_streak(matches: list, team_id: int) -> int:
    """Racha actual: positiva = victorias consecutivas, negativa = derrotas."""
    if not matches:
        return 0
    streak = 0
    first_result = None
    for m in matches:
        hs = m.get("home_score")
        aws = m.get("away_score")
        if hs is None or aws is None:
            continue
        is_home = m.get("home_team_id") == team_id
        if is_home:
            result = "W" if hs > aws else ("D" if hs == aws else "L")
        else:
            result = "W" if aws > hs else ("D" if hs == aws else "L")

        if first_result is None:
            first_result = result
        if result == first_result:
            streak += 1
        else:
            break

    if first_result == "W":
        return streak
    elif first_result == "L":
        return -streak
    return 0


def _avg_goals_total(matches: list) -> float:
    if not matches:
        return 0.0
    totals = [
        safe_int(m.get("home_score")) + safe_int(m.get("away_score"))
        for m in matches if m.get("home_score") is not None
    ]
    return np.mean(totals) if totals else 0.0


def _over_rate(matches: list, line: float) -> float:
    if not matches:
        return 0.0
    valid = [m for m in matches if m.get("home_score") is not None]
    if not valid:
        return 0.0
    over = sum(1 for m in valid if safe_int(m.get("home_score")) + safe_int(m.get("away_score")) > line)
    return over / len(valid)


def _avg_ht_goals(matches: list, team_id: int, is_home: bool) -> float:
    if not matches:
        return 0.0
    key = "home_ht_score" if is_home else "away_ht_score"
    vals = [safe_float(m.get(key)) for m in matches if m.get(key) is not None]
    return np.mean(vals) if vals else 0.0


def _days_since_last(matches: list, ref_date) -> int:
    if not matches:
        return 7
    last = matches[0]
    last_date = last.get("match_date")
    if isinstance(last_date, str):
        last_date = date.fromisoformat(last_date[:10])
    elif hasattr(last_date, "date"):
        last_date = last_date.date()
    if isinstance(ref_date, str):
        ref_date = date.fromisoformat(ref_date[:10])
    delta = (ref_date - last_date).days
    return max(0, delta)


# Feature names in order for models
MATCH_FEATURE_NAMES = [
    "home_win_rate_last10", "away_win_rate_last10",
    "home_goals_scored_avg", "away_goals_scored_avg",
    "home_goals_conceded_avg", "away_goals_conceded_avg",
    "home_corners_avg", "away_corners_avg",
    "home_shots_on_target_avg", "away_shots_on_target_avg",
    "h2h_home_wins", "h2h_draws", "h2h_avg_corners",
    "home_form_points", "away_form_points",
    "home_form_streak", "away_form_streak",
    "home_league_position", "away_league_position", "position_difference",
    "odds_home_win", "odds_draw", "odds_away_win",
    "odds_over_corners", "odds_implied_prob_home",
    "is_derby",
    "days_since_last_match_home", "days_since_last_match_away",
    "home_sentiment_score", "away_sentiment_score",
    "home_key_player_absent", "away_key_player_absent",
]

GOALS_FEATURE_NAMES = [
    "home_goals_scored_avg", "away_goals_scored_avg",
    "home_goals_conceded_avg", "away_goals_conceded_avg",
    "home_goals_avg_last5", "away_goals_avg_last5",
    "home_over25_rate", "away_over25_rate",
    "h2h_avg_goals",
    "home_ht_goals_avg", "away_ht_goals_avg",
    "home_form_points", "away_form_points",
    "home_win_rate_last10", "away_win_rate_last10",
    "position_difference", "is_derby",
    "odds_home_win", "odds_draw", "odds_away_win",
    "odds_implied_prob_home",
]

CORNERS_FEATURE_NAMES = [
    "home_corners_avg", "away_corners_avg", "h2h_avg_corners",
    "home_shots_on_target_avg", "away_shots_on_target_avg",
    "home_form_points", "away_form_points",
    "position_difference", "is_derby",
    "odds_over_corners",
    "home_league_position", "away_league_position",
]

PLAYER_FEATURE_NAMES = [
    "player_avg_sot", "player_avg_shots_total", "player_avg_minutes",
    "player_matches_count",
    "player_sot_last3", "player_minutes_last3",
    "opponent_goals_conceded_avg", "team_shots_on_target_avg",
    "is_derby", "is_home",
    "player_goals_last3",
]
