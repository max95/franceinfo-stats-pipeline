import os
import json
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import text
from db import get_engine
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except ImportError:  # pragma: no cover - fallback for very old python
    ZoneInfo = None

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TOK = int(os.getenv("SUMMARIZE_MAX_TOKENS", "800"))
WINDOW_DAYS = max(1, int(os.getenv("SUMMARY_WINDOW_DAYS", "3")))
MIN_HOUR = int(os.getenv("SUMMARY_MIN_HOUR", "9"))
TZ_NAME = os.getenv("SUMMARY_TIMEZONE")

PROMPT_DAILY = open("config/prompts/daily_summary.txt", "r", encoding="utf-8").read()
PROMPT_TOPIC = open("config/prompts/topic_summary.txt", "r", encoding="utf-8").read()

# Appel API compatible OpenAI (adaptable à autre fournisseur)
import requests
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")


def _now() -> datetime:
    """Return timezone-aware now when possible."""
    if TZ_NAME:
        if ZoneInfo is not None:
            try:
                return datetime.now(ZoneInfo(TZ_NAME))
            except Exception:
                pass
    if ZoneInfo is not None:
        try:
            return datetime.now().astimezone()
        except Exception:
            pass
    return datetime.now()


def call_llm(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0.3,
    }
    r = requests.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _parse_dt(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            if value.endswith("Z"):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return None
            return None
    return None


def _normalize_ts(value):
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


SQL_ARTICLES_BY_DAY = """
select title, summary, topic, published_at, inserted_at
from articles
where date(coalesce(published_at, inserted_at)) = :day
and id not in (select article_id from article_dupes)
order by coalesce(published_at, inserted_at)
"""

SQL_TOPICS_DAY = """
select distinct coalesce(topic, 'general') as topic
from articles
where date(coalesce(published_at, inserted_at)) = :day
  and id not in (select article_id from article_dupes)
"""


def iter_days(reference: date, now: datetime) -> list[date]:
    start = reference - timedelta(days=WINDOW_DAYS - 1)
    days = [start + timedelta(days=i) for i in range(WINDOW_DAYS)]
    if MIN_HOUR >= 0:
        cutoff = MIN_HOUR % 24
        if reference in days and now.hour < cutoff:
            days = [d for d in days if d != reference]
    return days


def build_daily(day: date):
    eng = get_engine()
    with eng.begin() as cxn:
        rows = cxn.execute(text(SQL_ARTICLES_BY_DAY), {"day": str(day)}).mappings().all()
        if not rows:
            return
        latest_ts = None
        for r in rows:
            candidates = [
                _parse_dt(r.get("published_at")),
                _parse_dt(r.get("inserted_at")),
            ]
            article_ts = max((ts for ts in candidates if ts), default=None)
            article_ts = _normalize_ts(article_ts)
            if article_ts and (latest_ts is None or article_ts > latest_ts):
                latest_ts = article_ts
        current = cxn.execute(
            text("select updated_at from daily_summaries where day = :d"),
            {"d": str(day)},
        ).mappings().one_or_none()
        if current and latest_ts:
            summary_ts = _normalize_ts(_parse_dt(current.get("updated_at")))
            if summary_ts and latest_ts <= summary_ts:
                return
        bundle = "\n".join([f"- {r['title']} — {r['summary'] or ''}" for r in rows])
        prompt = PROMPT_DAILY.format(date=str(day)) + "\n\nARTICLES:\n" + bundle
        out = call_llm(prompt)
        stamp = _now()
        cxn.execute(
            text(
                """
            insert into daily_summaries(day, summary_md, updated_at)
            values (:d, :s, :u)
            on conflict(day) do update set summary_md=excluded.summary_md, updated_at=excluded.updated_at
        """
            ),
            {"d": str(day), "s": out, "u": stamp},
        )


def build_topics(day: date):
    eng = get_engine()
    with eng.begin() as cxn:
        topics = [r[0] for r in cxn.execute(text(SQL_TOPICS_DAY), {"day": str(day)}).all()]
        for t in topics:
            rows = cxn.execute(text("""
                select title, summary, published_at, inserted_at
                from articles
                where date(coalesce(published_at, inserted_at)) = :day
                  and coalesce(topic,'general') = :t
                  and id not in (select article_id from article_dupes)
                order by coalesce(published_at, inserted_at)
            """), {"day": str(day), "t": t}).mappings().all()
            if not rows:
                continue
            latest_ts = None
            for r in rows:
                candidates = [
                    _parse_dt(r.get("published_at")),
                    _parse_dt(r.get("inserted_at")),
                ]
                article_ts = max((ts for ts in candidates if ts), default=None)
                article_ts = _normalize_ts(article_ts)
                if article_ts and (latest_ts is None or article_ts > latest_ts):
                    latest_ts = article_ts
            current = cxn.execute(
                text("select updated_at from topic_summaries where day = :d and topic = :t"),
                {"d": str(day), "t": t},
            ).mappings().one_or_none()
            if current and latest_ts:
                summary_ts = _normalize_ts(_parse_dt(current.get("updated_at")))
                if summary_ts and latest_ts <= summary_ts:
                    continue
            bundle = "\n".join([f"- {r['title']} — {r['summary'] or ''}" for r in rows])
            prompt = PROMPT_TOPIC.format(topic=t, date=str(day)) + "\n\nARTICLES:\n" + bundle
            out = call_llm(prompt)
            stamp = _now()
            cxn.execute(text("""
                insert into topic_summaries(day, topic, summary_md, updated_at)
                values (:d, :t, :s, :u)
                on conflict(day, topic) do update set summary_md=excluded.summary_md, updated_at=excluded.updated_at
            """), {"d": str(day), "t": t, "s": out, "u": stamp})


if __name__ == "__main__":
    now = _now()
    today = now.date()
    for day in iter_days(today, now):
        build_daily(day)
        build_topics(day)
