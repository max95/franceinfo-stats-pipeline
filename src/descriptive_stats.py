from __future__ import annotations

import argparse
import logging
from typing import Iterable, Sequence

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


def compute_theme_time_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "theme", "article_count"])

    counts = (
        df.groupby(["day", "theme"], dropna=False)
        .size()
        .reset_index(name="article_count")
        .sort_values(["day", "theme"])
    )
    return counts


def select_top_themes(time_series: pd.DataFrame, top_themes: int | None) -> pd.DataFrame:
    if time_series.empty or top_themes is None or top_themes <= 0:
        return time_series

    totals = (
        time_series.groupby("theme", dropna=False)["article_count"].sum()
        .sort_values(ascending=False)
    )
    selected = totals.head(top_themes).index
    return time_series[time_series["theme"].isin(selected)].reset_index(drop=True)


def compute_theme_correlations(
    time_series: pd.DataFrame, themes: Sequence[str] | None = None
) -> pd.DataFrame:
    if time_series.empty:
        return pd.DataFrame()

    pivot = (
        time_series.pivot_table(
            index="day",
            columns="theme",
            values="article_count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    if themes:
        missing = [theme for theme in themes if theme not in pivot.columns]
        if missing:
            LOG.warning("Thèmes introuvables pour la corrélation: %s", ", ".join(missing))
        selected_cols = [theme for theme in themes if theme in pivot.columns]
        if not selected_cols:
            return pd.DataFrame()
        pivot = pivot[selected_cols]

    if pivot.shape[1] < 2:
        LOG.warning("Impossible de calculer une corrélation avec moins de deux thèmes")
        return pd.DataFrame()

    corr = pivot.corr(method="pearson")
    return corr.round(3)


def detect_theme_peaks(
    time_series: pd.DataFrame,
    z_threshold: float = 2.5,
    min_days: int = 5,
) -> pd.DataFrame:
    if time_series.empty:
        return pd.DataFrame()

    stats = (
        time_series.groupby("theme")
        .agg(mean_count=("article_count", "mean"), std_count=("article_count", "std"), days=("day", "nunique"))
        .reset_index()
    )

    stats = stats[stats["days"] >= min_days]
    if stats.empty:
        LOG.warning(
            "Aucun thème n'a suffisamment de jours (%s) pour la détection de pics",
            min_days,
        )
        return pd.DataFrame()

    merged = time_series.merge(stats, on="theme", how="inner")
    merged = merged[merged["std_count"].fillna(0) > 0]
    if merged.empty:
        return pd.DataFrame()

    merged["z_score"] = (merged["article_count"] - merged["mean_count"]) / merged["std_count"]
    peaks = merged[merged["z_score"] >= z_threshold].copy()
    peaks.sort_values(["z_score", "article_count"], ascending=False, inplace=True)
    peaks["mean_count"] = peaks["mean_count"].round(2)
    peaks["std_count"] = peaks["std_count"].round(2)
    peaks["z_score"] = peaks["z_score"].round(2)
    return peaks[["day", "theme", "article_count", "mean_count", "std_count", "z_score"]]


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
    parser.add_argument(
        "--top-themes",
        dest="top_themes",
        type=int,
        default=10,
        help="Nombre de thèmes principaux à afficher pour les séries temporelles et corrélations",
    )
    parser.add_argument(
        "--correlate",
        nargs="+",
        help="Liste de thèmes pour calculer une matrice de corrélation ciblée",
    )
    parser.add_argument(
        "--z-threshold",
        dest="z_threshold",
        type=float,
        default=2.5,
        help="Seuil de détection des pics (z-score)",
    )
    parser.add_argument(
        "--min-days",
        dest="min_days",
        type=int,
        default=5,
        help="Nombre minimal de jours d'observation par thème pour analyser les pics",
    )
    args = parser.parse_args()

    df = fetch_articles(args.start_date, args.end_date)
    if df.empty:
        LOG.info("Aucun article trouvé sur la période demandée")
        return

    counts = compute_article_counts(df)
    weights = compute_theme_weights(df)
    lifespan_per_theme, overall = compute_subject_lifespan(df)
    full_time_series = compute_theme_time_series(df)
    theme_time_series = select_top_themes(full_time_series, args.top_themes)
    correlation_source = full_time_series if args.correlate else theme_time_series
    correlations = compute_theme_correlations(correlation_source, args.correlate)
    peaks = detect_theme_peaks(
        full_time_series,
        z_threshold=args.z_threshold,
        min_days=args.min_days,
    )

    display_table("Nombre d'articles par jour / source / thème", counts)
    display_table("Poids relatifs des thèmes", weights)
    display_table("Durée moyenne de vie médiatique par thème", lifespan_per_theme)
    display_table("Fréquence quotidienne par thème", theme_time_series)

    if not correlations.empty:
        print()
        print("Corrélations entre thèmes")
        print("-" * len("Corrélations entre thèmes"))
        print(correlations.to_string())
    elif args.correlate:
        LOG.info("Impossible de calculer la corrélation pour les thèmes demandés")

    display_table(
        "Pics médiatiques détectés (z-score)",
        peaks,
    )

    if overall is not None:
        print(f"\nDurée moyenne de vie médiatique tous thèmes confondus : {overall} jours")


if __name__ == "__main__":
    main()
