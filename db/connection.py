from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import DATABASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_connection():
    return engine.connect()


def test_connection():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Conexion a BD exitosa")
        return True
    except Exception as e:
        logger.error(f"Error conectando a BD: {e}")
        return False
