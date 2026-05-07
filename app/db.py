from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import cfg
# DB Starts
engine = create_engine(
    cfg.DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in cfg.DB_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
