# db.py
# ----------------------------------------------------------
# Shared Database Connection (SQLAlchemy + psycopg2)
# ----------------------------------------------------------

import os
from sqlalchemy import create_engine
import psycopg2
from urllib.parse import urlparse

# Get the DB URL from environment or fallback to local
DB_URL = os.getenv("DATABASE_URL", "postgresql://greer_user:@localhost:5432/yfinance_db")

def get_engine():
    """
    Returns a SQLAlchemy engine for use with pandas or ORM queries.
    """
    return create_engine(DB_URL)

def get_psycopg_connection():
    """
    Returns a raw psycopg2 connection.
    Parses DATABASE_URL if needed.
    """
    url = urlparse(DB_URL)
    return psycopg2.connect(
        dbname=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port or 5432
    )
