import os
import json
from datetime import date, timedelta
from sqlalchemy import text
from db import get_engine
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TOK = int(os.getenv("SUMMARIZE_MAX_TOKENS", "800"))

PROMPT_DAILY = open("config/prompts/daily_summary.txt", "r", encoding="utf-8").read()
PROMPT_TOPIC = open("config/prompts/topic_summary.txt", "r", encoding="utf-8").read()

# Appel API compatible OpenAI (adaptable à autre fournisseur)
import requests
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")


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


SQL_ARTICLES_BY_DAY = """
select title, summary, topic
from articles
where date(coalesce(published_at, inserted_at)) = :day
and id not in (select article_id from article_dupes)
order by coalesce(published_at, inserted_at)
"""

SQL_TOPICS_DAY = """
select distinct coalesce(topic, 'general') as topic
from articles
where date(coalesce(published_at, inserted_at)) = :day
"""


def build_daily(day: date):
    eng = get_engine()
    with eng.begin() as cxn:
        rows = cxn.execute(text(SQL_ARTICLES_BY_DAY), {"day": str(day)}).mappings().all()
        if not rows:
            return
        bundle = "\n".join([f"- {r['title']} — {r['summary'] or ''}" for r in rows])
        prompt = PROMPT_DAILY.format(date=str(day)) + "\n\nARTICLES:\n" + bundle
        out = call_llm(prompt)
        cxn.execute(text("""
            insert into daily_summaries(day, summary_md)
            values (:d, :s)
            on conflict(day) do update set summary_md=excluded.summary_md
        """), {"d": str(day), "s": out})


def build_topics(day: date):
    eng = get_engine()
    with eng.begin() as cxn:
        topics = [r[0] for r in cxn.execute(text(SQL_TOPICS_DAY), {"day": str(day)}).all()]
        for t in topics:
            rows = cxn.execute(text("""
                select title, summary from articles
                where date(coalesce(published_at, inserted_at)) = :day
                  and coalesce(topic,'general') = :t
                  and id not in (select article_id from article_dupes)
                order by coalesce(published_at, inserted_at)
            """), {"day": str(day), "t": t}).mappings().all()
            if not rows:
                continue
            bundle = "\n".join([f"- {r['title']} — {r['summary'] or ''}" for r in rows])
            prompt = PROMPT_TOPIC.format(topic=t, date=str(day)) + "\n\nARTICLES:\n" + bundle
            out = call_llm(prompt)
            cxn.execute(text("""
                insert into topic_summaries(day, topic, summary_md)
                values (:d, :t, :s)
                on conflict(day, topic) do update set summary_md=excluded.summary_md
            """), {"d": str(day), "t": t, "s": out})


if __name__ == "__main__":
    today = date.today()
    build_daily(today)
    build_topics(today)
