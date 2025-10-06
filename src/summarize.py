#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize France Info articles by day and topic
"""

import os
import json
import requests
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import text
from db import get_engine
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo  # Python ≥ 3.9
except ImportError:
    ZoneInfo = None

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TOK = int(os.getenv("SUMMARIZE_MAX_TOKENS", "800"))
WINDOW_DAYS = max(1, int(os.getenv("SUMMARY_WINDOW_DAYS", "3")))
MIN_HOUR = int(os.getenv("SUMMARY_MIN_HOUR", "9"))
TZ_NAME = os.getenv("SUMMARY_TIMEZONE", "Europe/Paris")

PROMPT_DAILY = open("config/prompts/daily_summary.txt", "r", encoding="utf-8").read()
PROMPT_TOPIC = open("config/prompts/topic_summary.txt", "r", encoding="utf-8").read()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def _now() -> datetime:
    """Return timezone-aware now when possible."""
    if TZ_NAME and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(TZ_NAME))
        except Exception:
            pass
    try:
        return datetime.now().astimezone()
    except Exception:
        return datetime.now()


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
    """Convert datetime to UTC-naive for safe comparison."""
    if value is None:
        return None
    if value.tzinfo is None:
        # Interprète les naïves comme TZ_NAME si dispo
        if TZ_NAME and ZoneInfo is not None:
            value = value.replace(tzinfo=ZoneInfo(TZ_NAME))
        else:
            return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _latest_ts_from_row(row, cols=("published_at", "inserted_at")):
    """Pick the latest timestamp among given columns, normalized to UTC-naive."""
    vals = []
    for c in cols:
        ts = _parse_dt(row.get(c))
        ts = _normalize_ts(ts)
        if ts:
            vals.append(ts)
    return max(vals) if vals else None


def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0.3,
    }
    r = requests.post(
        f"{OPENAI_API_BASE}/chat/completions",
        json=payload,
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------
# SQL QUERIES
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------

def iter_days(reference: date, now: datetime) -> list[date]:
    """Return list of days to summarize (sliding window)."""
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
            article_ts = _latest_ts_from_row(r)
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
            on conflict(day) do update
              set summary_md=excluded.summary_md, updated_at=excluded.updated_at
        """
            ),
            {"d": str(day), "s": out, "u": stamp},
        )


def build_topics(day: date):
    eng = get_engine()
    with eng.begin() as cxn:
        topics = [r[0] for r in cxn.execute(text(SQL_TOPICS_DAY), {"day": str(day)}).all()]
        for t in topics:
            rows = cxn.execute(
                text("""
                    select title, summary, published_at, inserted_at
                    from articles
                    where date(coalesce(published_at, inserted_at)) = :day
                      and coalesce(topic,'general') = :t
                      and id not in (select article_id from article_dupes)
                    order by coalesce(published_at, inserted_at)
                """),
                {"day": str(day), "t": t},
            ).mappings().all()

            if not rows:
                continue

            latest_ts = None
            for r in rows:
                article_ts = _latest_ts_from_row(r)
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

            cxn.execute(
                text("""
                    insert into topic_summaries(day, topic, summary_md, updated_at)
                    values (:d, :t, :s, :u)
                    on conflict(day, topic) do update
                      set summary_md=excluded.summary_md, updated_at=excluded.updated_at
                """),
                {"d": str(day), "t": t, "s": out, "u": stamp},
            )

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    now = _now()
    today = now.date()
    for day in iter_days(today, now):
        build_daily(day)
        build_topics(day)
