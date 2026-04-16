"""
Re-entrenamiento diario, disparado automáticamente cuando se consolidan
los resultados del día. Usa cache en memoria para velocidad.
"""
from datetime import datetime
from scripts.train_cached import build_caches, install_patches
from models.trainer import train_all_models
from models.evaluator import should_promote_model, adjust_threshold, get_model_version
from db.models import get_config
from notifications.telegram import send_telegram
from utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_retrain():
    """Flujo completo de re-entrenamiento diario con cache."""
    logger.info("=== INICIANDO RE-ENTRENAMIENTO DIARIO ===")

    try:
        # Cache en memoria para velocidad
        cache = build_caches()
        install_patches(cache)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version = f"v{timestamp}"

        old_versions = {
            "1x2": get_model_version("1x2"),
            "corners": get_model_version("corners"),
            "player_shots": get_model_version("player_shots"),
        }

        results = train_all_models(version)

        promotions = {}
        for model_type, metrics in results.items():
            if "error" in metrics:
                promotions[model_type] = False
                logger.warning(f"Modelo {model_type}: {metrics['error']}")
                continue
            promote = should_promote_model(metrics, model_type)
            promotions[model_type] = promote

        adjust_threshold()

        # Telegram
        acc_1x2 = results.get("1x2", {}).get("accuracy_cv", 0)
        samples = results.get("1x2", {}).get("training_samples", "?")

        report_lines = [
            f"\U0001f504 *Re-entrenamiento diario*",
            f"\U0001f4e6 Version: `{version}`",
            "",
        ]

        for model_type, metrics in results.items():
            if "error" in metrics:
                report_lines.append(f"\u26a0\ufe0f *{model_type}*: {metrics['error']}")
                continue
            promoted = promotions.get(model_type, False)
            status = "\u2705 Promovido" if promoted else "\u274c Mantenido anterior"
            if model_type == "1x2":
                report_lines.append(
                    f"\u26bd *1X2*: Acc={metrics['accuracy_cv']:.1%}, "
                    f"F1={metrics['f1_score']:.3f} | {status}"
                )
            else:
                r2 = metrics.get("r2_cv", 0)
                mae = metrics.get("mae_cv", 0)
                report_lines.append(
                    f"\U0001f4ca *{model_type}*: R2={r2:.3f}, "
                    f"MAE={mae:.2f} | {status}"
                )

        threshold = get_config("current_threshold") or "0.60"
        report_lines.extend([
            "",
            f"\U0001f3af Umbral: {float(threshold):.0%} | Muestras: {samples}",
            f"\U0001f4c5 Próxima predicción: mañana 6 AM",
        ])

        send_telegram("\n".join(report_lines))

        logger.info("=== RE-ENTRENAMIENTO COMPLETADO ===")
        return results

    except Exception as e:
        logger.error(f"Error en re-entrenamiento diario: {e}")
        send_telegram(f"\u26a0\ufe0f Error en re-entrenamiento: {str(e)[:200]}")
        return {}
