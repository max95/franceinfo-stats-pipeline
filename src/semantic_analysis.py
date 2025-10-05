"""Semantic analysis utilities for the news monitoring pipeline.

This module provides:
- automatic keyword extraction per time period,
- detection of emerging expressions/concepts,
- clustering of similar articles.

It can be executed as a CLI for ad-hoc analyses.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

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


if TYPE_CHECKING:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer


_TFIDF_VECTOR_CLS: type["TfidfVectorizer"] | None = None
_KMEANS_CLS: type["KMeans"] | None = None


def _require_tfidf_vectorizer() -> type["TfidfVectorizer"]:
    global _TFIDF_VECTOR_CLS
    if _TFIDF_VECTOR_CLS is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as _Cls
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "scikit-learn est requis pour calculer les TF-IDF. "
                "Installe les dépendances avec `pip install -r requirements.txt`."
            ) from exc
        _TFIDF_VECTOR_CLS = _Cls
    return _TFIDF_VECTOR_CLS


def _require_kmeans() -> type["KMeans"]:
    global _KMEANS_CLS
    if _KMEANS_CLS is None:
        try:
            from sklearn.cluster import KMeans as _Cls
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "scikit-learn est requis pour effectuer le clustering K-Means. "
                "Installe les dépendances avec `pip install -r requirements.txt`."
            ) from exc
        _KMEANS_CLS = _Cls
    return _KMEANS_CLS


@dataclass
class KeywordPeriod:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    keywords: dict[str, float]


def fetch_articles(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Fetch article metadata and textual content from the database."""

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
        "    a.title,",
        "    coalesce(a.summary, '') as summary",
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
    LOG.debug("Executing semantic analysis query: %s", query)

    with engine.begin() as cxn:
        rows = cxn.execute(text(query), params).mappings().all()

    if not rows:
        return pd.DataFrame(
            columns=["article_id", "day", "source", "theme", "title", "summary"]
        )

    df = pd.DataFrame(
        rows,
        columns=["article_id", "day", "source", "theme", "title", "summary"],
    )
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day"]).assign(day=lambda d: d["day"].dt.tz_localize(None))
    df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna(""))
    return df.dropna(subset=["text"]).assign(text=lambda d: d["text"].str.strip())


def _vectorizer(max_features: int | None, *, min_df: int = 2):
    vectorizer_cls = _require_tfidf_vectorizer()
    return vectorizer_cls(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="french",
        strip_accents="unicode",
        min_df=min_df,
    )


def compute_periodic_keywords(
    df: pd.DataFrame,
    *,
    period: str = "W",
    top_n: int = 15,
    max_features: int | None = 5000,
) -> list[KeywordPeriod]:
    """Compute dominant keywords for each time period."""

    if df.empty:
        return []

    df = df.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day", "text"])
    df = df[df["text"].str.len() > 0]

    if df.empty:
        return []

    df = df.sort_values("day")
    grouped = df.set_index("day").groupby(pd.Grouper(freq=period))

    periods: list[KeywordPeriod] = []
    for period_start, group in grouped:
        if group.empty:
            continue

        min_df = 1 if len(group) < 2 else 2
        vectorizer = _vectorizer(max_features, min_df=min_df)
        try:
            matrix = vectorizer.fit_transform(group["text"].tolist())
        except ValueError:
            LOG.debug("Skipping period %s: vocabulary is empty", period_start)
            continue
        feature_names = vectorizer.get_feature_names_out()
        weights = matrix.sum(axis=0).A1
        top_idx = weights.argsort()[::-1][:top_n]
        keywords = {feature_names[i]: float(weights[i]) for i in top_idx if weights[i] > 0}
        period_start_dt = group.index.min().normalize()
        period_end_dt = group.index.max().normalize()
        periods.append(
            KeywordPeriod(
                period_start=period_start_dt,
                period_end=period_end_dt,
                keywords=keywords,
            )
        )

    return periods


def detect_emerging_expressions(
    periods: Sequence[KeywordPeriod],
    *,
    min_score: float = 0.2,
    surge_ratio: float = 3.0,
    top_per_period: int = 5,
) -> pd.DataFrame:
    """Identify new or surging expressions based on keyword weights."""

    if not periods:
        return pd.DataFrame(columns=["period_start", "term", "score", "status"])

    seen_terms: set[str] = set()
    max_score_by_term: dict[str, float] = defaultdict(float)
    records: list[dict[str, object]] = []

    for idx, period in enumerate(periods):
        sorted_terms = sorted(period.keywords.items(), key=lambda kv: kv[1], reverse=True)
        taken = 0
        for term, score in sorted_terms:
            if score < min_score:
                continue

            status: str | None = None
            previous_best = max_score_by_term.get(term)
            if term not in seen_terms:
                status = "nouveau"
            elif previous_best > 0 and score / previous_best >= surge_ratio:
                status = "en hausse"

            max_score_by_term[term] = max(max_score_by_term[term], score)

            if status:
                records.append(
                    {
                        "period_start": period.period_start,
                        "term": term,
                        "score": round(score, 3),
                        "status": status,
                    }
                )
                taken += 1
                if taken >= top_per_period:
                    break

        seen_terms.update(period.keywords.keys())

    return pd.DataFrame(records)


def cluster_articles(
    df: pd.DataFrame,
    *,
    n_clusters: int = 8,
    max_features: int | None = 8000,
    random_state: int = 0,
    top_terms: int = 10,
) -> pd.DataFrame:
    """Cluster articles based on their textual content."""

    if df.empty:
        return pd.DataFrame(columns=["cluster", "article_ids", "top_terms"])

    df = df[df["text"].str.len() > 0]
    if df.empty:
        return pd.DataFrame(columns=["cluster", "article_ids", "top_terms"])

    n_clusters = max(1, min(n_clusters, len(df)))

    min_df = 1 if len(df) < 2 else 2
    vectorizer = _vectorizer(max_features, min_df=min_df)
    try:
        matrix = vectorizer.fit_transform(df["text"].tolist())
    except ValueError:
        return pd.DataFrame(columns=["cluster", "article_ids", "top_terms"])
    feature_names = vectorizer.get_feature_names_out()

    if n_clusters == 1:
        labels = [0] * len(df)
    else:
        kmeans_cls = _require_kmeans()
        model = kmeans_cls(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = model.fit_predict(matrix)

    df = df.assign(cluster=labels)

    rows: list[dict[str, object]] = []
    for cluster_id, group in df.groupby("cluster"):
        indices = group.index.tolist()
        cluster_matrix = matrix[indices]
        centroid = cluster_matrix.mean(axis=0).A1
        top_idx = centroid.argsort()[::-1][:top_terms]
        top_words = [feature_names[i] for i in top_idx if centroid[i] > 0]
        rows.append(
            {
                "cluster": int(cluster_id),
                "article_ids": group["article_id"].tolist(),
                "top_terms": top_words,
            }
        )

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def _format_keyword_period(period: KeywordPeriod) -> str:
    start_str = period.period_start.strftime("%Y-%m-%d")
    end_str = period.period_end.strftime("%Y-%m-%d")
    top_terms = ", ".join(f"{term} ({score:.2f})" for term, score in period.keywords.items())
    return f"[{start_str} → {end_str}] {top_terms}"


def run_cli(args: argparse.Namespace) -> None:
    df = fetch_articles(args.start_date, args.end_date)
    LOG.info("Fetched %d articles", len(df))

    periods = compute_periodic_keywords(
        df,
        period=args.period,
        top_n=args.top_keywords,
        max_features=args.max_features,
    )
    if not periods:
        LOG.warning("No articles found for the given parameters.")
        return

    LOG.info("Keywords par période (%s):", args.period)
    for period in periods:
        LOG.info(_format_keyword_period(period))

    emerging = detect_emerging_expressions(
        periods,
        min_score=args.min_score,
        surge_ratio=args.surge_ratio,
        top_per_period=args.emerging_per_period,
    )
    if emerging.empty:
        LOG.info("Aucune nouvelle expression détectée avec les seuils fournis.")
    else:
        LOG.info("Expressions émergentes ou en forte hausse:")
        for row in emerging.itertuples(index=False):
            LOG.info("%s — %s (%.3f)", row.period_start.date(), row.term, row.score)

    clusters = cluster_articles(
        df,
        n_clusters=args.clusters,
        max_features=args.cluster_max_features,
        random_state=args.random_state,
        top_terms=args.cluster_top_terms,
    )
    LOG.info("Clustering réalisé en %d groupes", len(clusters))
    for row in clusters.itertuples(index=False):
        LOG.info("Cluster %d (%d articles): %s", row.cluster, len(row.article_ids), ", ".join(row.top_terms))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse sémantique des articles.")
    parser.add_argument("--start-date", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Date de fin (YYYY-MM-DD)")
    parser.add_argument(
        "--period",
        default="W",
        help="Granularité temporelle pour les mots-clés (ex: D, W, M).",
    )
    parser.add_argument(
        "--top-keywords", type=int, default=15, help="Nombre de mots-clés principaux par période."
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Nombre maximum de termes pour le calcul TF-IDF.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="Score TF-IDF minimal pour considérer une expression comme émergente.",
    )
    parser.add_argument(
        "--surge-ratio",
        type=float,
        default=3.0,
        help="Facteur d'augmentation requis pour marquer un terme comme en forte hausse.",
    )
    parser.add_argument(
        "--emerging-per-period",
        type=int,
        default=5,
        help="Nombre maximum d'expressions émergentes à retourner par période.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Nombre de clusters à produire pour le regroupement d'articles.",
    )
    parser.add_argument(
        "--cluster-max-features",
        type=int,
        default=8000,
        help="Nombre maximum de termes pour le clustering.",
    )
    parser.add_argument(
        "--cluster-top-terms",
        type=int,
        default=10,
        help="Nombre de termes saillants à afficher par cluster.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Graine aléatoire pour le clustering K-Means.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_cli(args)
    except ModuleNotFoundError as exc:
        LOG.error("%s", exc)
        LOG.error(
            "Assure-toi d'installer les dépendances avec `pip install -r requirements.txt`."
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
