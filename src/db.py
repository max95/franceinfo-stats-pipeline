import os
from pathlib import Path

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


def _ensure_article_content_hash(engine: Engine):
    with engine.begin() as cxn:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            existing = {
                row["name"]
                for row in cxn.execute(text("pragma table_info(articles)")).mappings()
            }
            if "content_hash" not in existing:
                cxn.execute(text("alter table articles add column content_hash text"))
        else:
            has_column = cxn.execute(
                text(
                    """
                    select 1
                    from information_schema.columns
                    where table_name = :table
                      and column_name = :column
                      and table_schema = current_schema()
                    limit 1
                    """
                ),
                {"table": "articles", "column": "content_hash"},
            ).first()
            if not has_column:
                cxn.execute(text("alter table articles add column content_hash text"))


def init_schema():
    engine = get_engine()
    kind = "postgres" if DSN.startswith("postgresql") else "sqlite"
    schema_path = Path("sql") / f"schema_{'postgres' if kind == 'postgres' else 'sqlite'}.sql"
    with engine.begin() as cxn:
        sql = schema_path.read_text(encoding="utf-8")
        for stmt in [s for s in sql.split(";\n") if s.strip()]:
            cxn.execute(text(stmt))

    _ensure_article_content_hash(engine)

if __name__ == "__main__":
    init_schema()
