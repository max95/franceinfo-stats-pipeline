import json
import logging
import ssl
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone
import hashlib
import yaml
import feedparser
from sqlalchemy import text
from dotenv import load_dotenv

# SSL CA
try:
    import certifi
    CERT_FILE = certifi.where()
except Exception:
    CERT_FILE = None  # on tombera sur le store système

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
LOG = logging.getLogger(__name__)

CFG = yaml.safe_load(open("config/rss_sources.yml", "r", encoding="utf-8"))

UA = "franceinfo-stats-pipeline/1.0 (+https://github.com/jeromebouchet)"
TIMEOUT = 15  # secondes

def _ssl_context():
    ctx = ssl.create_default_context(cafile=CERT_FILE) if CERT_FILE else ssl.create_default_context()
    # pas d’insecure ici !
    return ctx

def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context()))
    with opener.open(req, timeout=TIMEOUT) as resp:
        return resp.read()

def _parse_feed(url: str, source_name: str):
    """Télécharge le flux avec certs valides, puis parse avec feedparser."""
    try:
        raw = _fetch_bytes(url)
    except HTTPError as e:
        LOG.warning("HTTP %s sur %s (%s)", e.code, source_name, url)
        raise
    except URLError as e:
        LOG.warning("URLError sur %s (%s): %s", source_name, url, e)
        raise
    except ssl.SSLError as e:
        LOG.error("Erreur SSL persistante sur %s (%s): %s", source_name, url, e)
        raise

    feed = feedparser.parse(raw)
    if getattr(feed, "bozo", False) and getattr(feed, "bozo_exception", None):
        LOG.warning("Flux %s bozo=%s: %s", source_name, feed.bozo, feed.bozo_exception)
    return feed

def _parse_entry_datetime(entry) -> datetime | None:
    # 1) champs *parsed* de feedparser (struct_time)
    for k in ("published_parsed", "updated_parsed", "created_parsed"):
        st = getattr(entry, k, None)
        if st:
            return datetime(*st[:6], tzinfo=timezone.utc)
    # 2) champs texte — essayons email.utils
    from email.utils import parsedate_to_datetime
    for k in ("published", "updated", "created"):
        v = entry.get(k)
        if v:
            try:
                dt = parsedate_to_datetime(v)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
    # 3) ISO 8601 best-effort (certaines sources)
    for k in ("published", "updated", "created"):
        v = entry.get(k)
        if v:
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except Exception:
                continue
    return None

def _guid(entry) -> str:
    cand = entry.get("id") or entry.get("guid") or entry.get("link") or ""
    if not cand:
        # hash minimaliste du contenu brut
        body = json.dumps({k: entry.get(k) for k in entry.keys()}, ensure_ascii=False, sort_keys=True)
        cand = hashlib.sha1(body.encode("utf-8")).hexdigest()
    return str(cand)

def upsert_sources():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            cxn.execute(text("""
                insert into sources(name, url, topic)
                values(:name, :url, :topic)
                on conflict(name)
                do update set url = excluded.url, topic = excluded.topic
            """), s)

def ingest():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            LOG.info("Fetching feed %s", s["name"])
            try:
                feed = _parse_feed(s["url"], s["name"])
            except Exception:
                # on a déjà loggé le détail
                continue

            for e in feed.entries:
                dt = _parse_entry_datetime(e)
                raw = json.dumps({k: e.get(k) for k in e.keys()}, ensure_ascii=False)

                params = {
                    "guid": _guid(e),
                    "link": e.get("link"),
                    "title": e.get("title"),
                    "summary": e.get("summary"),
                    "published_at": dt,
                    "topic": s.get("topic"),
                    "raw": raw,
                    "src": s["name"],
                }

                try:
                    cxn.execute(text("""
                        insert into articles(source_id, guid, link, title, summary, published_at, topic, raw)
                        select id, :guid, :link, :title, :summary, :published_at, :topic, :raw
                        from sources where name = :src
                        on conflict do nothing
                    """), params)
                except Exception as exc:
                    LOG.exception("Insertion échouée pour %s (%s): %s", s["name"], e.get("link", ""), exc)

if __name__ == "__main__":
    from db import get_engine  # import ici pour éviter les imports cycles
    upsert_sources()
    ingest()
