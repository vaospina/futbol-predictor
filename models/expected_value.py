"""
Calculo de Expected Value (EV) para apuestas.
"""


def expected_value(prob_win: float, odds: float) -> float:
    """
    EV = (prob_win * (odds - 1)) - (1 - prob_win)

    EV > 0 -> apuesta con valor (+EV)
    EV < 0 -> apuesta sin valor (-EV)

    Ejemplo: prob=0.70, odds=1.80
    EV = (0.70 * 0.80) - 0.30 = 0.56 - 0.30 = +0.26 (26% de valor)
    """
    if odds <= 0 or prob_win <= 0:
        return -1.0
    return (prob_win * (odds - 1)) - (1 - prob_win)


def roi_from_predictions(predictions: list, stake: float = 10000) -> float:
    """
    Calcula ROI simulado con stake fijo.
    predictions: lista de dicts con 'result' y 'odds'
    """
    if not predictions:
        return 0.0

    total_staked = len(predictions) * stake
    total_return = 0.0

    for p in predictions:
        if p.get("result") == "win":
            total_return += stake * (p.get("odds", 1.0))
        # Si pierde, return = 0

    roi = ((total_return - total_staked) / total_staked) * 100 if total_staked > 0 else 0.0
    return round(roi, 2)
