"""
Evaluador de modelos: compara versiones, decide si promover.
Ajuste dinamico de umbral segun rendimiento.
"""
from db.models import (
    get_config, set_config, get_active_model,
    get_cumulative_stats, fetch_one,
)
from config.settings import MIN_THRESHOLD, MAX_THRESHOLD
from utils.logger import get_logger

logger = get_logger(__name__)


def should_promote_model(new_metrics: dict, model_type: str) -> bool:
    """
    Decide si el nuevo modelo debe reemplazar al activo.
    Compara accuracy_cv del nuevo vs el activo.
    """
    active = get_active_model(model_type)
    if not active:
        logger.info(f"No hay modelo activo de tipo {model_type}, promoviendo nuevo")
        return True

    old_accuracy = active.get("accuracy_cv") or 0
    new_accuracy = new_metrics.get("accuracy_cv") or new_metrics.get("r2_cv", 0)

    if new_accuracy >= old_accuracy:
        logger.info(
            f"Nuevo modelo {model_type} mejor: {new_accuracy:.3f} >= {old_accuracy:.3f}"
        )
        return True
    else:
        logger.info(
            f"Modelo {model_type} anterior es mejor: {old_accuracy:.3f} > {new_accuracy:.3f}"
        )
        return False


def adjust_threshold():
    """
    Ajusta el umbral dinamico basado en el rendimiento acumulado.
    - Si accuracy > 75%: subir umbral 2%
    - Si accuracy < 65%: bajar umbral 2%
    """
    stats = get_cumulative_stats()
    total = stats.get("total", 0) if stats else 0
    correct = stats.get("correct", 0) if stats else 0

    if total < 10:
        logger.info("Menos de 10 predicciones, manteniendo umbral actual")
        return

    accuracy = correct / total
    current_threshold = float(get_config("current_threshold") or 0.60)

    old_threshold = current_threshold
    if accuracy > 0.75:
        current_threshold = min(current_threshold + 0.02, MAX_THRESHOLD)
    elif accuracy < 0.65:
        current_threshold = max(current_threshold - 0.02, MIN_THRESHOLD)

    if current_threshold != old_threshold:
        set_config("current_threshold", str(round(current_threshold, 2)))
        logger.info(
            f"Umbral ajustado: {old_threshold:.2f} -> {current_threshold:.2f} "
            f"(accuracy: {accuracy:.2%})"
        )
    else:
        logger.info(
            f"Umbral mantenido: {current_threshold:.2f} (accuracy: {accuracy:.2%})"
        )


def get_current_threshold() -> float:
    return float(get_config("current_threshold") or 0.60)


def get_model_version(model_type: str) -> str:
    key_map = {
        "1x2": "model_active_1x2",
        "corners": "model_active_corners",
        "player_shots": "model_active_shots",
    }
    return get_config(key_map.get(model_type, "")) or "v1.0"
