from datetime import datetime, timedelta
import pytz
from config.settings import TIMEZONE


def now_colombia():
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz)


def today_colombia():
    return now_colombia().date()


def date_range(start_date, end_date):
    delta = end_date - start_date
    for i in range(delta.days + 1):
        yield start_date + timedelta(days=i)


def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default
