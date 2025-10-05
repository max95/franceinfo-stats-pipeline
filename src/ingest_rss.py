import json
import logging
import yaml
import feedparser
from datetime import datetime
from sqlalchemy import text
from dotenv import load_dotenv
from db import get_engine

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
LOG = logging.getLogger(__name__)

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
            LOG.info("Fetching feed %s", s["name"])
            try:
                feed = feedparser.parse(s["url"])  # pas de requêtes externes lourdes
            except Exception as exc:
                LOG.exception("Impossible de parser le flux %s (%s)", s["name"], s["url"])
                continue

            if getattr(feed, "bozo", False) and getattr(feed, "bozo_exception", None):
                LOG.warning(
                    "Flux %s renvoyé avec erreurs: %s",
                    s["name"],
                    feed.bozo_exception,
                )

            for e in feed.entries:
                published = e.get("published") or e.get("updated")
                if published:
                    try:
                        # feedparser parse au format struct_time parfois
                        if hasattr(e, "published_parsed") and e.published_parsed:
                            dt = datetime(*e.published_parsed[:6])
                        else:
                            dt = datetime.fromisoformat(published)
                    except Exception as exc:
                        LOG.warning(
                            "Date intraitable '%s' pour %s (%s): %s",
                            published,
                            s["name"],
                            getattr(e, "link", ""),
                            exc,
                        )
                        dt = None
                else:
                    dt = None
                raw = json.dumps({k: str(getattr(e, k, "")) for k in e.keys()}, ensure_ascii=False)
                try:
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
                except Exception as exc:
                    LOG.exception(
                        "Insertion échouée pour %s (%s): %s",
                        s["name"],
                        getattr(e, "link", ""),
                        exc,
                    )

if __name__ == "__main__":
    upsert_sources()
    ingest()
