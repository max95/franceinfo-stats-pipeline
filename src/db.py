import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DB_DSN", "sqlite:///franceinfo.db")

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DSN, future=True)
    return _engine


def init_schema():
    engine = get_engine()
    kind = "postgres" if DSN.startswith("postgresql") else "sqlite"
    schema_path = f"sql/schema_{'postgres' if kind=='postgres' else 'sqlite'}.sql"
    with engine.begin() as cxn:
        sql = open(schema_path, "r", encoding="utf-8").read()
        for stmt in [s for s in sql.split(";\n") if s.strip()]:
            cxn.execute(text(stmt))

if __name__ == "__main__":
    init_schema()
