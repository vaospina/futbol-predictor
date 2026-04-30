# Ligas domesticas (predicciones diarias)
# season: API-Football season number. European leagues use academic year (2025 = 2025-26).
# LATAM calendar-year leagues use the actual year (2026).
# Liga MX quirk: Clausura 2026 (Jan-Jun) is season 2025 on the API (Apertura start year).
DOMESTIC_LEAGUES = {
    39:  {"name": "Premier League",       "country": "Inglaterra", "season": 2025},
    135: {"name": "Serie A",              "country": "Italia",     "season": 2025},
    78:  {"name": "Bundesliga",           "country": "Alemania",   "season": 2025},
    140: {"name": "La Liga",              "country": "España",     "season": 2025},
    61:  {"name": "Ligue 1",              "country": "Francia",    "season": 2025},
    88:  {"name": "Eredivisie",           "country": "Holanda",    "season": 2025},
    71:  {"name": "Brasileirao Serie A",  "country": "Brasil",     "season": 2026},
    262: {"name": "Liga MX",              "country": "Mexico",     "season": 2025},
    128: {"name": "Argentine Primera",    "country": "Argentina",  "season": 2026},
}

# Copas (predicciones cuando haya partidos)
CUP_LEAGUES = {
    2:  {"name": "UEFA Champions League", "country": "Europa",     "season": 2025},
    3:  {"name": "UEFA Europa League",    "country": "Europa",     "season": 2025},
    13: {"name": "Copa Libertadores",     "country": "Sudamerica", "season": 2026},
    11: {"name": "Copa Sudamericana",     "country": "Sudamerica", "season": 2026},
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
    ("Chelsea", "Tottenham"), ("Tottenham", "Chelsea"),
    # Serie A
    ("AC Milan", "Inter"), ("Inter", "AC Milan"),
    ("Juventus", "Torino"), ("Torino", "Juventus"),
    ("AS Roma", "Lazio"), ("Lazio", "AS Roma"),
    ("Napoli", "Juventus"), ("Juventus", "Napoli"),
    # Bundesliga
    ("Borussia Dortmund", "Schalke 04"), ("Schalke 04", "Borussia Dortmund"),
    ("Bayern Munich", "Borussia Dortmund"), ("Borussia Dortmund", "Bayern Munich"),
    ("Hamburger SV", "Werder Bremen"), ("Werder Bremen", "Hamburger SV"),
    # La Liga
    ("Real Madrid", "Barcelona"), ("Barcelona", "Real Madrid"),
    ("Real Madrid", "Atletico Madrid"), ("Atletico Madrid", "Real Madrid"),
    ("Barcelona", "Espanyol"), ("Espanyol", "Barcelona"),
    ("Sevilla", "Real Betis"), ("Real Betis", "Sevilla"),
    ("Athletic Club", "Real Sociedad"), ("Real Sociedad", "Athletic Club"),
    # Ligue 1
    ("Paris Saint Germain", "Olympique Marseille"), ("Olympique Marseille", "Paris Saint Germain"),
    ("Olympique Lyonnais", "Olympique Marseille"), ("Olympique Marseille", "Olympique Lyonnais"),
    ("Monaco", "Nice"), ("Nice", "Monaco"),
    # Eredivisie
    ("Ajax", "Feyenoord"), ("Feyenoord", "Ajax"),
    ("Ajax", "PSV Eindhoven"), ("PSV Eindhoven", "Ajax"),
    ("Feyenoord", "PSV Eindhoven"), ("PSV Eindhoven", "Feyenoord"),
    # Brasileirao
    ("Flamengo", "Fluminense"), ("Fluminense", "Flamengo"),
    ("Corinthians", "Palmeiras"), ("Palmeiras", "Corinthians"),
    ("Sao Paulo", "Corinthians"), ("Corinthians", "Sao Paulo"),
    ("Flamengo", "Vasco da Gama"), ("Vasco da Gama", "Flamengo"),
    ("Atletico Mineiro", "Cruzeiro"), ("Cruzeiro", "Atletico Mineiro"),
    # Liga MX
    ("America", "Chivas"), ("Chivas", "America"),
    ("America", "Cruz Azul"), ("Cruz Azul", "America"),
    ("Chivas", "Atlas"), ("Atlas", "Chivas"),
    ("Pumas UNAM", "America"), ("America", "Pumas UNAM"),
    # Argentine Primera
    ("Boca Juniors", "River Plate"), ("River Plate", "Boca Juniors"),
    ("Racing Club", "Independiente"), ("Independiente", "Racing Club"),
    ("San Lorenzo", "Huracan"), ("Huracan", "San Lorenzo"),
}
