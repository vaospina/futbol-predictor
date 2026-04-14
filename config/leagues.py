# Ligas domesticas (predicciones diarias)
DOMESTIC_LEAGUES = {
    39: {"name": "Premier League", "country": "Inglaterra"},
    135: {"name": "Serie A", "country": "Italia"},
    78: {"name": "Bundesliga", "country": "Alemania"},
    71: {"name": "Brasileirao Serie A", "country": "Brasil"},
}

# Copas (predicciones cuando haya partidos)
CUP_LEAGUES = {
    2: {"name": "UEFA Champions League", "country": "Europa"},
    13: {"name": "Copa Libertadores", "country": "Sudamerica"},
}

# Todas las ligas combinadas
ALL_LEAGUES = {**DOMESTIC_LEAGUES, **CUP_LEAGUES}

# Derbies conocidos (para feature is_derby)
DERBIES = {
    # Premier League
    ("Arsenal", "Tottenham"), ("Tottenham", "Arsenal"),
    ("Liverpool", "Everton"), ("Everton", "Liverpool"),
    ("Manchester United", "Manchester City"), ("Manchester City", "Manchester United"),
    ("Chelsea", "Arsenal"), ("Arsenal", "Chelsea"),
    # Serie A
    ("AC Milan", "Inter"), ("Inter", "AC Milan"),
    ("Juventus", "Torino"), ("Torino", "Juventus"),
    ("AS Roma", "Lazio"), ("Lazio", "AS Roma"),
    # Bundesliga
    ("Borussia Dortmund", "Schalke 04"), ("Schalke 04", "Borussia Dortmund"),
    ("Bayern Munich", "Borussia Dortmund"), ("Borussia Dortmund", "Bayern Munich"),
    # Brasileirao
    ("Flamengo", "Fluminense"), ("Fluminense", "Flamengo"),
    ("Corinthians", "Palmeiras"), ("Palmeiras", "Corinthians"),
    ("Sao Paulo", "Corinthians"), ("Corinthians", "Sao Paulo"),
}
