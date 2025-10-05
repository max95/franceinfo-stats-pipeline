#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import date
from sqlalchemy import text
from db import get_engine
import gradio as gr

DAY = date.today().isoformat()

SQL_SUMMARY = "select day, summary_md from daily_summaries where day = :d"
SQL_TOPICS = "select topic, summary_md from topic_summaries where day = :d order by topic"
SQL_STATS = """
select
  (select count(*) from articles where date(coalesce(published_at, inserted_at)) = :d) as articles,
  (select count(*) from article_dupes) as dupes,
  (select count(*) from transcripts where day = :d) as transcripts,
  (select count(*) from crosslinks where day = :d) as crosslinks
"""
SQL_FEED = """
select a.id, s.name as source, coalesce(a.topic,'general') as topic, a.title,
       date(coalesce(a.published_at, a.inserted_at)) as day, a.link
from articles a join sources s on s.id=a.source_id
where date(coalesce(a.published_at, a.inserted_at)) = :d
  and a.id not in (select article_id from article_dupes)
order by coalesce(a.published_at, a.inserted_at) desc
limit 200
"""


def load_data(day: str):
    eng = get_engine()
    with eng.begin() as cxn:
        sumrow = cxn.execute(text(SQL_SUMMARY), {"d": day}).mappings().first()
        topicrows = cxn.execute(text(SQL_TOPICS), {"d": day}).mappings().all()
        stats = cxn.execute(text(SQL_STATS), {"d": day}).mappings().first()
        feed = pd.read_sql_query(text(SQL_FEED), cxn, params={"d": day})
    summary_md = sumrow["summary_md"] if sumrow else "*(Pas de résumé pour ce jour)*"
    topics_md = "\n\n".join([f"### {r['topic'].title()}\n\n" + r['summary_md'] for r in topicrows]) or ""
    return summary_md, topics_md, stats, feed


def ui_refresh(day):
    sm, tm, st, feed = load_data(day)
    stats_md = f"**Articles**: {st['articles']} | **Doublons**: {st['dupes']} | **Transcripts**: {st['transcripts']} | **Crosslinks**: {st['crosslinks']}"
    return sm, tm, stats_md, feed

with gr.Blocks(title="France Info — RSS x Transcripts", theme=gr.themes.Soft()) as demo:
    day = gr.State(DAY)

    gr.Markdown("# France Info — RSS × Transcripts (Jour)")
    stats_box = gr.Markdown()
    daily_box = gr.Markdown()
    topics_box = gr.Markdown()
    feed_tbl = gr.Dataframe(interactive=False)

    def _init():
        sm, tm, st, feed = load_data(day.value)
        stats_box.value = f"**Articles**: {st['articles']} | **Doublons**: {st['dupes']} | **Transcripts**: {st['transcripts']} | **Crosslinks**: {st['crosslinks']}"
        daily_box.value = sm
        topics_box.value = tm
        feed_tbl.value = feed

    demo.load(_init)

    def _tick():
        sm, tm, st, feed = ui_refresh(day.value)
        return sm, tm, st, feed

    timer = gr.Timer(10.0, True)
    timer.tick(fn=_tick, outputs=[daily_box, topics_box, stats_box, feed_tbl])

if __name__ == "__main__":
    demo.launch()
