import json
import logging
import ssl
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone
import hashlib
import yaml
import feedparser
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text
from dotenv import load_dotenv

from db import get_engine, init_schema

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
ARTICLE_TIMEOUT = 20  # secondes


def _dt_to_iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return None


def _compute_content_hash(title: str | None, summary: str | None) -> str | None:
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if summary and summary.strip():
        parts.append(summary.strip())
    if not parts:
        return None
    joined = "\n\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

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


def _clean_text(text: str | None) -> str | None:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned or None


def _extract_main_text(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        tag.decompose()

    for tag_name in ("article", "main"):
        node = soup.find(tag_name)
        if node:
            text = _clean_text(node.get_text("\n", strip=True))
            if text:
                return text

    best_text = None
    best_len = 0
    for node in soup.find_all(["section", "div"]):
        text = _clean_text(node.get_text("\n", strip=True))
        if text:
            length = len(text)
            if length > best_len:
                best_text = text
                best_len = length

    if best_text:
        return best_text

    return _clean_text(soup.get_text("\n", strip=True))


def _fetch_article_content(url: str) -> str | None:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
            timeout=ARTICLE_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOG.warning("Article introuvable %s: %s", url, exc)
        return None

    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype:
        return None

    return _extract_main_text(resp.text)

def upsert_sources():
    init_schema()
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
    init_schema()
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            LOG.info("Fetching feed %s", s["name"])
            try:
                feed = _parse_feed(s["url"], s["name"])
            except Exception:
                # on a déjà loggé le détail
                continue

            source_id = cxn.execute(
                text("select id from sources where name = :name"), {"name": s["name"]}
            ).scalar_one()

            for e in feed.entries:
                dt = _parse_entry_datetime(e)
                raw = json.dumps({k: e.get(k) for k in e.keys()}, ensure_ascii=False)

                guid = _guid(e) or None
                link = e.get("link")
                summary = e.get("summary") or e.get("description")
                content_hash = _compute_content_hash(e.get("title"), summary)

                published_iso = _dt_to_iso(dt)

                existing = None
                candidates: dict[int, dict] = {}
                if guid:
                    for row in cxn.execute(
                        text(
                            "select id, content, content_hash, published_at "
                            "from articles where source_id = :source_id and guid = :guid"
                        ),
                        {"source_id": source_id, "guid": guid},
                    ).mappings():
                        candidates[row["id"]] = dict(row)
                if link:
                    for row in cxn.execute(
                        text(
                            "select id, content, content_hash, published_at "
                            "from articles where source_id = :source_id and link = :link"
                        ),
                        {"source_id": source_id, "link": link},
                    ).mappings():
                        candidates[row["id"]] = dict(row)

                if candidates:
                    for cand in candidates.values():
                        cand_iso = _dt_to_iso(cand.get("published_at"))
                        if cand_iso == published_iso or (cand_iso is None and published_iso is None):
                            existing = cand
                            break
                    if not existing:
                        existing = next(iter(candidates.values()))

                if existing and content_hash and existing.get("content_hash") == content_hash:
                    LOG.debug("Article %s déjà présent (hash identique), skip", link or guid)
                    continue

                content = None
                if link and (not existing or existing.get("content") is None or existing.get("content_hash") != content_hash):
                    content = _fetch_article_content(link)

                dialect = cxn.dialect.name
                published_for_db = dt
                if dialect == "sqlite" and dt is not None:
                    published_for_db = dt.isoformat()
                now = datetime.now(timezone.utc)
                updated_at = now if dialect != "sqlite" else now.isoformat()

                params = {
                    "guid": guid,
                    "link": link,
                    "title": e.get("title"),
                    "summary": summary,
                    "content": content,
                    "published_at": published_for_db,
                    "topic": s.get("topic"),
                    "raw": raw,
                    "source_id": source_id,
                    "content_hash": content_hash,
                    "updated_at": updated_at,
                }

                try:
                    if existing:
                        cxn.execute(
                            text(
                                """
                                update articles
                                set link = :link,
                                    title = :title,
                                    summary = :summary,
                                    published_at = :published_at,
                                    topic = :topic,
                                    raw = :raw,
                                    content = coalesce(:content, content),
                                    content_hash = :content_hash,
                                    updated_at = :updated_at
                                where id = :id
                                """
                            ),
                            {**params, "id": existing["id"]},
                        )
                    else:
                        cxn.execute(
                            text(
                                """
                                insert into articles(
                                    source_id, guid, link, title, summary, content,
                                    published_at, topic, raw, content_hash, updated_at
                                )
                                values(:source_id, :guid, :link, :title, :summary, :content,
                                       :published_at, :topic, :raw, :content_hash, :updated_at)
                                """
                            ),
                            params,
                        )
                except Exception as exc:
                    LOG.exception("Insertion échouée pour %s (%s): %s", s["name"], e.get("link", ""), exc)

if __name__ == "__main__":
    upsert_sources()
    ingest()
