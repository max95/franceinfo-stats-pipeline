import json
import logging
import ssl
import urllib.request
from urllib.error import URLError

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


def _parse_feed(url: str, source_name: str):
    """Parse a feed while retrying with relaxed SSL verification if needed."""
    feed = feedparser.parse(url)

    bozo_exc = getattr(feed, "bozo_exception", None)
    if getattr(feed, "bozo", False) and bozo_exc:
        if isinstance(bozo_exc, ssl.SSLError):
            ssl_err = bozo_exc
        elif isinstance(bozo_exc, URLError) and isinstance(getattr(bozo_exc, "reason", None), ssl.SSLError):
            ssl_err = bozo_exc.reason
        else:
            ssl_err = None

        if ssl_err is not None:
            LOG.warning(
                "Flux %s renvoyé avec une erreur SSL (%s); nouvel essai sans vérification",
                source_name,
                ssl_err,
            )
            insecure_context = ssl.create_default_context()
            insecure_context.check_hostname = False
            insecure_context.verify_mode = ssl.CERT_NONE
            feed = feedparser.parse(url, handlers=[urllib.request.HTTPSHandler(context=insecure_context)])

    return feed


def ingest():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            LOG.info("Fetching feed %s", s["name"])
            try:
                feed = _parse_feed(s["url"], s["name"])  # pas de requêtes externes lourdes
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
