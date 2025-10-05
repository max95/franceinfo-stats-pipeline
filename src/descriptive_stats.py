from __future__ import annotations

import argparse
import logging
from typing import Iterable

import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from db import get_engine, init_schema  # type: ignore
else:
    from .db import get_engine, init_schema


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
LOG = logging.getLogger(__name__)

DATE_SQL = "date(coalesce(a.published_at, a.inserted_at))"


def _mode(values: Iterable[str]) -> str:
    series = pd.Series(list(values))
    modes = series.mode(dropna=True)
    if modes.empty:
        first = series.dropna()
        return first.iloc[0] if not first.empty else "inconnu"
    return modes.iloc[0]


def fetch_articles(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    init_schema()
    engine = get_engine()
    sql = [
        "select",
        "    a.id as article_id,",
        f"    {DATE_SQL} as day,",
        "    coalesce(s.name, 'inconnu') as source,",
        "    coalesce((",
        "        select t.theme",
        "        from article_themes t",
        "        where t.article_id = a.id",
        "          and (t.content_hash = a.content_hash or t.content_hash is null)",
        "        order by t.confidence desc",
        "        limit 1",
        "    ), a.topic, 'inconnu') as theme,",
        "    a.content_hash",
        "from articles a",
        "left join sources s on s.id = a.source_id",
    ]

    conditions: list[str] = [
        "coalesce(a.published_at, a.inserted_at) is not null",
    ]
    params: dict[str, str] = {}

    if start_date:
        conditions.append(f"{DATE_SQL} >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append(f"{DATE_SQL} <= :end_date")
        params["end_date"] = end_date

    if conditions:
        sql.append("where " + " and ".join(conditions))

    sql.append(f"order by {DATE_SQL}, source")

    query = "\n".join(sql)
    with engine.begin() as cxn:
        rows = cxn.execute(text(query), params).mappings().all()

    if not rows:
        return pd.DataFrame(columns=["article_id", "day", "source", "theme", "content_hash"])

    df = pd.DataFrame(rows, columns=["article_id", "day", "source", "theme", "content_hash"])
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["day"]).assign(day=lambda d: d["day"].dt.date)


def compute_article_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["day", "source", "theme"], dropna=False)
        .size()
        .reset_index(name="article_count")
        .sort_values(["day", "source", "theme"])
    )
    return grouped


def compute_theme_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df.groupby("theme", dropna=False).size().reset_index(name="article_count")
    total = counts["article_count"].sum()
    counts["share_pct"] = (counts["article_count"] / total * 100).round(2)
    return counts.sort_values("article_count", ascending=False)


def compute_subject_lifespan(df: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    subjects = df.dropna(subset=["content_hash"])
    if subjects.empty:
        return pd.DataFrame(columns=["theme", "avg_span_days", "median_span_days", "subjects"]), None

    grouped = subjects.groupby("content_hash")
    spans = grouped.agg(
        first_day=("day", "min"),
        last_day=("day", "max"),
        theme=("theme", _mode),
    )
    spans[["first_day", "last_day"]] = spans[["first_day", "last_day"]].apply(
        pd.to_datetime, errors="coerce"
    )
    spans = spans.dropna(subset=["first_day", "last_day"])
    if spans.empty:
        return (
            pd.DataFrame(columns=["theme", "avg_span_days", "median_span_days", "subjects"]),
            None,
        )

    spans["span_days"] = (spans["last_day"] - spans["first_day"]).dt.days + 1

    per_theme = (
        spans.groupby("theme")
        .agg(
            avg_span_days=("span_days", "mean"),
            median_span_days=("span_days", "median"),
            subjects=("span_days", "size"),
        )
        .reset_index()
        .sort_values("avg_span_days", ascending=False)
    )
    per_theme["avg_span_days"] = per_theme["avg_span_days"].round(2)
    per_theme["median_span_days"] = per_theme["median_span_days"].round(2)

    overall = spans["span_days"].mean()
    overall_mean = round(float(overall), 2) if pd.notna(overall) else None
    return per_theme, overall_mean


def display_table(title: str, df: pd.DataFrame) -> None:
    print()
    print(title)
    print("-" * len(title))
    if df.empty:
        print("(aucune donnée)")
        return
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistiques descriptives sur les articles")
    parser.add_argument("--start-date", dest="start_date", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end-date", dest="end_date", help="Date de fin (YYYY-MM-DD)")
    args = parser.parse_args()

    df = fetch_articles(args.start_date, args.end_date)
    if df.empty:
        LOG.info("Aucun article trouvé sur la période demandée")
        return

    counts = compute_article_counts(df)
    weights = compute_theme_weights(df)
    lifespan_per_theme, overall = compute_subject_lifespan(df)

    display_table("Nombre d'articles par jour / source / thème", counts)
    display_table("Poids relatifs des thèmes", weights)
    display_table("Durée moyenne de vie médiatique par thème", lifespan_per_theme)

    if overall is not None:
        print(f"\nDurée moyenne de vie médiatique tous thèmes confondus : {overall} jours")


if __name__ == "__main__":
    main()
