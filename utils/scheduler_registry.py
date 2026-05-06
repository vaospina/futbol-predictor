"""
Registro global del scheduler para permitir que flujos internos agreguen jobs dinámicos.
Se registra desde main.py al arrancar y se consulta desde los flujos que necesitan
programar tareas (ej: verificación de lineups 60 min antes del partido).
"""
_scheduler = None


def register(scheduler):
    global _scheduler
    _scheduler = scheduler


def get_scheduler():
    return _scheduler
