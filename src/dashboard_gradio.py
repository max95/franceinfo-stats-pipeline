#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import date, timedelta
import os

import gradio as gr
import pandas as pd
import plotly.express as px
from sqlalchemy import text

from db import get_engine

DAY = date.today().isoformat()

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2026-01-31"
DEFAULT_PERIOD = "W"
DEFAULT_CLUSTERS = "10"

def _format_dash_config(config: dict[str, str]) -> str:
    start = config.get("start_date") or "(inconnue)"
    end = config.get("end_date") or "(inconnue)"
    period = config.get("period") or "(non défini)"
    clusters = config.get("clusters") or "(non défini)"
    return (
        "### Paramètres de l'analyse\n"
        f"- **Période étudiée** : {start} → {end}\n"
        f"- **Granularité temporelle** : {period}\n"
        f"- **Nombre de clusters sémantiques** : {clusters}"
    )


DASH_CONFIG = {
    "start_date": os.getenv("DASH_START_DATE", DEFAULT_START_DATE),
    "end_date": os.getenv("DASH_END_DATE", DEFAULT_END_DATE),
    "period": os.getenv("DASH_PERIOD", DEFAULT_PERIOD),
    "clusters": os.getenv("DASH_CLUSTERS", DEFAULT_CLUSTERS),
}

CONFIG_MD = _format_dash_config(DASH_CONFIG)

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
SQL_TOPIC_TRENDS = """
select date(coalesce(a.published_at, a.inserted_at)) as day,
       s.name as source,
       coalesce(a.topic,'general') as topic,
       count(*) as articles
from articles a
join sources s on s.id = a.source_id
where date(coalesce(a.published_at, a.inserted_at)) between :start_day and :end_day
  and a.id not in (select article_id from article_dupes)
group by
  day,
  source,
  coalesce(a.topic,'general')
order by
  day asc,
  source asc,
  coalesce(a.topic,'general') asc
"""


def _build_topic_trend_visuals(
    trends: pd.DataFrame,
    selected_sources: list[str] | None,
    selected_topics: list[str] | None,
):
    df = trends.copy()
    selected_sources = selected_sources or []
    selected_topics = selected_topics or []
    if selected_sources:
        df = df[df["source"].isin(selected_sources)]
    if selected_topics:
        df = df[df["topic"].isin(selected_topics)]
    df = df.sort_values(["day", "source", "topic"])
    if df.empty:
        return None, df

    df_plot = df.copy()
    df_plot["day"] = pd.to_datetime(df_plot["day"]).dt.strftime("%Y-%m-%d")
    if "share_pct" not in df_plot:
        df_plot["share_pct"] = df_plot["share"] * 100
    df_plot["share_pct"] = df_plot["share_pct"].round(2)
    facet_args: dict[str, str] = {}
    if df_plot["source"].nunique() > 1:
        facet_args["facet_row"] = "source"

    fig = px.bar(
        df_plot,
        x="day",
        y="share_pct",
        color="topic",
        barmode="stack",
        category_orders={"day": sorted(df_plot["day"].unique())},
        hover_data={"share_pct": True, "articles": True},
        labels={"share_pct": "Part (%)", "articles": "Articles", "day": "Jour"},
        **facet_args,
    )
    fig.update_layout(
        title="Répartition des thèmes par source (30 derniers jours)",
        xaxis_title="Jour",
        yaxis_title="Part des articles (%)",
        legend_title_text="Thème",
        bargap=0.05,
    )
    fig.update_yaxes(range=[0, 100])
    fig.update_xaxes(type="category", tickangle=45)
    df_table = df.copy()
    if "share" in df_table:
        df_table = df_table.drop(columns=["share"])
    if "share_pct" in df_table:
        df_table = df_table.rename(columns={"share_pct": "part_pct"})
        df_table["part_pct"] = df_table["part_pct"].round(2)
    return fig, df_table.reset_index(drop=True)


def load_data(day: str):
    eng = get_engine()
    start_day = (date.fromisoformat(day) - timedelta(days=29)).isoformat()
    with eng.begin() as cxn:
        sumrow = cxn.execute(text(SQL_SUMMARY), {"d": day}).mappings().first()
        topicrows = cxn.execute(text(SQL_TOPICS), {"d": day}).mappings().all()
        stats = cxn.execute(text(SQL_STATS), {"d": day}).mappings().first()
        feed = pd.read_sql_query(text(SQL_FEED), cxn, params={"d": day})
        topic_trends = pd.read_sql_query(
            text(SQL_TOPIC_TRENDS),
            cxn,
            params={"start_day": start_day, "end_day": day},
        )
    if not topic_trends.empty:
        totals = topic_trends.groupby(["day", "source"])["articles"].transform("sum")
        topic_trends["share"] = topic_trends["articles"] / totals
        topic_trends["share_pct"] = (topic_trends["share"] * 100).round(2)
        topic_trends = topic_trends.sort_values(["day", "source", "topic"])
    else:
        topic_trends = topic_trends.assign(
            share=pd.Series(dtype="float64"),
            share_pct=pd.Series(dtype="float64"),
        )
    summary_md = sumrow["summary_md"] if sumrow else "*(Pas de résumé pour ce jour)*"
    topics_md = "\n\n".join([f"### {r['topic'].title()}\n\n" + r['summary_md'] for r in topicrows]) or ""
    sources = sorted(topic_trends["source"].unique().tolist()) if not topic_trends.empty else []
    topics = sorted(topic_trends["topic"].unique().tolist()) if not topic_trends.empty else []
    return summary_md, topics_md, stats, feed, topic_trends, sources, topics


def ui_refresh(day, selected_sources, selected_topics):
    (
        sm,
        tm,
        stats,
        feed,
        topic_trends,
        sources,
        topics,
    ) = load_data(day)
    stats_md = (
        f"**Articles**: {stats['articles']} | **Doublons**: {stats['dupes']} | "
        f"**Transcripts**: {stats['transcripts']} | **Crosslinks**: {stats['crosslinks']}"
    )
    fig, table = _build_topic_trend_visuals(topic_trends, selected_sources, selected_topics)
    return sm, tm, stats_md, feed, topic_trends, fig, table, sources, topics


with gr.Blocks(title="France Info — RSS x Transcripts", theme=gr.themes.Soft()) as demo:
    day = gr.State(DAY)
    topic_trends_state = gr.State()

    gr.Markdown("# France Info — RSS × Transcripts (Jour)")
    gr.Markdown(CONFIG_MD)
    stats_box = gr.Markdown()
    daily_box = gr.Markdown()
    topics_box = gr.Markdown()
    feed_tbl = gr.Dataframe(interactive=False)

    gr.Markdown("## Thèmes & sujets par source")
    with gr.Row():
        source_filter = gr.Dropdown(label="Sources", multiselect=True)
        topic_filter = gr.Dropdown(label="Thèmes", multiselect=True)
    trends_plot = gr.Plot()
    trends_table = gr.Dataframe(interactive=False)

    def _init():
        (
            sm,
            tm,
            stats_md,
            feed,
            topic_trends,
            sources,
            topics,
        ) = ui_refresh(day.value, None, None)
        stats_box.value = stats_md
        daily_box.value = sm
        topics_box.value = tm
        feed_tbl.value = feed
        topic_trends_state.value = topic_trends
        source_filter.choices = sources
        topic_filter.choices = topics
        source_filter.value = None
        topic_filter.value = None
        fig, table = _build_topic_trend_visuals(topic_trends, None, None)
        trends_plot.value = fig
        trends_table.value = table

    demo.load(_init)

    def _tick(current_day, selected_sources, selected_topics):
        (
            sm,
            tm,
            stats_md,
            feed,
            topic_trends,
            fig,
            table,
            sources,
            topics,
        ) = ui_refresh(current_day, selected_sources, selected_topics)
        selected_sources = selected_sources or []
        selected_topics = selected_topics or []
        selected_sources = [s for s in selected_sources if s in sources]
        selected_topics = [t for t in selected_topics if t in topics]
        topic_trends_state_value = topic_trends
        return (
            sm,
            tm,
            stats_md,
            feed,
            topic_trends_state_value,
            fig,
            table,
            gr.update(choices=sources, value=selected_sources or None),
            gr.update(choices=topics, value=selected_topics or None),
        )

    timer = gr.Timer(10.0, True)
    timer.tick(
        fn=_tick,
        inputs=[day, source_filter, topic_filter],
        outputs=[
            daily_box,
            topics_box,
            stats_box,
            feed_tbl,
            topic_trends_state,
            trends_plot,
            trends_table,
            source_filter,
            topic_filter,
        ],
    )

    def _on_filter_change(selected_sources, selected_topics, topic_trends):
        fig, table = _build_topic_trend_visuals(topic_trends, selected_sources, selected_topics)
        return fig, table

    source_filter.change(
        fn=_on_filter_change,
        inputs=[source_filter, topic_filter, topic_trends_state],
        outputs=[trends_plot, trends_table],
    )
    topic_filter.change(
        fn=_on_filter_change,
        inputs=[source_filter, topic_filter, topic_trends_state],
        outputs=[trends_plot, trends_table],
    )

if __name__ == "__main__":
    demo.launch()
