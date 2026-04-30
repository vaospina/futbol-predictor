# Ligas domesticas (predicciones diarias)
DOMESTIC_LEAGUES = {
    39:  {"name": "Premier League",       "country": "Inglaterra"},
    135: {"name": "Serie A",              "country": "Italia"},
    78:  {"name": "Bundesliga",           "country": "Alemania"},
    140: {"name": "La Liga",              "country": "España"},
    61:  {"name": "Ligue 1",              "country": "Francia"},
    88:  {"name": "Eredivisie",           "country": "Holanda"},
    71:  {"name": "Brasileirao Serie A",  "country": "Brasil"},
    262: {"name": "Liga MX",              "country": "Mexico"},
    128: {"name": "Argentine Primera",    "country": "Argentina"},
}

# Copas (predicciones cuando haya partidos)
CUP_LEAGUES = {
    2:  {"name": "UEFA Champions League", "country": "Europa"},
    3:  {"name": "UEFA Europa League",    "country": "Europa"},
    13: {"name": "Copa Libertadores",     "country": "Sudamerica"},
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
