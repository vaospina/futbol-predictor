"""
Re-entrenamiento diario, disparado automáticamente cuando se consolidan
los resultados del día. Usa cache en memoria para velocidad.
"""
from datetime import datetime
from scripts.train_cached import build_caches, install_patches
from models.trainer import train_all_models
from models.evaluator import should_promote_model, adjust_threshold, get_model_version
from db.models import get_config, get_active_model, fetch_one
from notifications.telegram import send_telegram
from utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_retrain():
    """Flujo completo de re-entrenamiento diario con cache."""
    logger.info("=== INICIANDO RE-ENTRENAMIENTO DIARIO ===")

    try:
        # Obtener métricas del modelo activo ANTES de reentrenar (para el diff de accuracy)
        old_model = get_active_model("1x2") or {}
        old_acc = old_model.get("accuracy_cv") or 0

        max_date_row = fetch_one(
            "SELECT TO_CHAR(MAX(match_date AT TIME ZONE 'America/Bogota'), 'YYYY-MM-DD')"
            " AS max_date FROM matches WHERE status='finished'"
        )
        max_date_str = (max_date_row.get("max_date") or "?") if max_date_row else "?"

        # Cache en memoria para velocidad
        cache = build_caches()
        install_patches(cache)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version = f"v{timestamp}"

        results = train_all_models(version)

        promotions = {}
        for model_type, metrics in results.items():
            if "error" in metrics or "skipped" in metrics:
                promotions[model_type] = False
                reason = metrics.get("error") or metrics.get("reason", "skipped")
                logger.warning(f"Modelo {model_type}: {reason}")
                continue
            promote = should_promote_model(metrics, model_type)
            promotions[model_type] = promote

        adjust_threshold()

        # Construir reporte Telegram
        new_acc = results.get("1x2", {}).get("accuracy_cv", 0)
        samples = results.get("1x2", {}).get("training_samples", "?")
        samples_fmt = f"{samples:,}" if isinstance(samples, int) else str(samples)

        any_success = any("accuracy_cv" in m or "r2_cv" in m for m in results.values())
        all_failed = all("error" in m for m in results.values())

        if all_failed:
            status_str = "❌ Fallido"
        elif any_success:
            status_str = "✅ Exitoso"
        else:
            status_str = "⚠️ Parcial (shots saltado)"

        report_lines = [
            f"\U0001f9e0 *Reentrenamiento*: {status_str}",
            f"\U0001f4ca Datos: {samples_fmt} partidos (último: {max_date_str})",
        ]

        if new_acc and old_acc:
            report_lines.append(f"\U0001f3af Accuracy CV: {old_acc:.1%} → {new_acc:.1%}")
        elif new_acc:
            report_lines.append(f"\U0001f3af Accuracy CV: {new_acc:.1%}")

        report_lines.append(f"\U0001f4c8 Versión: `{version}`")
        report_lines.append("")

        for model_type, metrics in results.items():
            if "error" in metrics:
                report_lines.append(
                    f"⚠️ *{model_type}*: {metrics['error']} "
                    f"({metrics.get('samples', '?')} muestras)"
                )
            elif "skipped" in metrics:
                reason = metrics.get("reason", "saltado").replace("<", "").replace(">", "")
                report_lines.append(f"⏭️ *{model_type}*: {reason}")
            else:
                promoted = promotions.get(model_type, False)
                promo_str = "✅ Promovido" if promoted else "❌ Mantenido anterior"
                if model_type == "1x2":
                    report_lines.append(
                        f"⚽ *1X2*: Acc={metrics['accuracy_cv']:.1%}, "
                        f"F1={metrics['f1_score']:.3f} | {promo_str}"
                    )
                else:
                    r2 = metrics.get("r2_cv", 0)
                    mae = metrics.get("mae_cv", 0)
                    report_lines.append(
                        f"\U0001f4ca *{model_type}*: R2={r2:.3f}, MAE={mae:.2f} | {promo_str}"
                    )

        threshold = get_config("current_threshold") or "0.60"
        report_lines.extend([
            "",
            f"\U0001f3af Umbral: {float(threshold):.0%} | \U0001f4c5 Próxima predicción: mañana 6 AM",
        ])

        send_telegram("\n".join(report_lines))

        logger.info("=== RE-ENTRENAMIENTO COMPLETADO ===")
        return results

    except Exception as e:
        logger.error(f"Error en re-entrenamiento diario: {e}")
        send_telegram(f"⚠️ Error en re-entrenamiento: {str(e)[:200]}")
        return {}
