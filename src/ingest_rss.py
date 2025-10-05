import json
import yaml
import feedparser
from datetime import datetime
from pathlib import Path
from sqlalchemy import text
from dotenv import load_dotenv
from db import get_engine

load_dotenv()
CFG = yaml.safe_load(open("config/rss_sources.yml", "r", encoding="utf-8"))

def upsert_sources():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            cxn.execute(text(
                """
                insert into sources(name, url, topic)
                values(:name, :url, :topic)
                on conflict(name) do update set url=excluded.url, topic=excluded.topic
                """
            ), s)


def ingest():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            feed = feedparser.parse(s["url"])  # pas de requêtes externes lourdes
            for e in feed.entries:
                published = e.get("published") or e.get("updated")
                if published:
                    try:
                        # feedparser parse au format struct_time parfois
                        if hasattr(e, "published_parsed") and e.published_parsed:
                            dt = datetime(*e.published_parsed[:6])
                        else:
                            dt = datetime.fromisoformat(published)
                    except Exception:
                        dt = None
                else:
                    dt = None
                raw = json.dumps({k: str(getattr(e, k, "")) for k in e.keys()}, ensure_ascii=False)
                cxn.execute(text(
                    """
                    insert into articles(source_id, guid, link, title, summary, published_at, topic, raw)
                    select id, :guid, :link, :title, :summary, :published_at, :topic, :raw
                    from sources where name=:src
                    on conflict do nothing
                    """
                ), {
                    "guid": getattr(e, "id", None),
                    "link": e.link,
                    "title": e.title,
                    "summary": getattr(e, "summary", None),
                    "published_at": dt,
                    "topic": s.get("topic"),
                    "raw": raw,
                    "src": s["name"],
                })

if __name__ == "__main__":
    upsert_sources()
    ingest()
