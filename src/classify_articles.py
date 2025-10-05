from __future__ import annotations

import argparse
import logging
from typing import Callable

import requests
from sqlalchemy import text
from dotenv import load_dotenv

from db import get_engine, init_schema
from source_classifiers import Article, pick_classifier

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
LOG = logging.getLogger(__name__)

UA = "franceinfo-stats-pipeline/1.0 (+https://github.com/jeromebouchet)"
HTML_TIMEOUT = 20


def _make_html_fetcher() -> Callable[[str], str | None]:
    cache: dict[str, str | None] = {}

    def fetch_html(url: str) -> str | None:
        if url in cache:
            return cache[url]

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
                timeout=HTML_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOG.warning("Échec de récupération HTML %s: %s", url, exc)
            cache[url] = None
            return None

        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype:
            cache[url] = None
            return None

        cache[url] = resp.text
        return cache[url]

    return fetch_html


def fetch_articles(engine, since: str | None) -> list[dict]:
    with engine.begin() as cxn:
        conditions = ["a.content_hash is not null"]
        params: dict[str, str] = {}
        if since:
            conditions.append("date(coalesce(a.published_at, a.inserted_at)) >= :since")
            params["since"] = since
        conditions.append(
            "not exists (select 1 from article_themes t where t.article_id = a.id and t.content_hash = a.content_hash)"
        )
        sql = """
            select
                a.id,
                a.title,
                a.summary,
                a.content,
                a.content_hash,
                a.link,
                coalesce(s.name, '') as source_name,
                a.raw
            from articles a
            left join sources s on s.id = a.source_id
        """
        if conditions:
            sql += " where " + " and ".join(conditions)
        sql += " order by coalesce(a.published_at, a.inserted_at)"
        rows = cxn.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]


def process_articles(since: str | None = None) -> None:
    init_schema()
    eng = get_engine()
    to_classify = fetch_articles(eng, since)
    if not to_classify:
        LOG.info("Aucun article à classifier")
        return

    fetch_html = _make_html_fetcher()

    with eng.begin() as cxn:
        for art in to_classify:
            article = Article(
                id=art["id"],
                source_name=art.get("source_name") or "",
                link=art.get("link"),
                title=art.get("title"),
                summary=art.get("summary"),
                content=art.get("content"),
                content_hash=art.get("content_hash"),
                raw=art.get("raw"),
            )

            if not article.content_hash:
                LOG.debug("Article %s sans content_hash, skip", article.id)
                continue

            cxn.execute(
                text("delete from article_themes where article_id = :id and content_hash != :hash"),
                {"id": article.id, "hash": article.content_hash},
            )

            classifier = pick_classifier(article)
            if not classifier:
                LOG.info(
                    "Aucun classifieur configuré pour la source '%s' (article %s)",
                    article.source_name or "inconnue",
                    article.id,
                )
                continue

            results = classifier.classify(article, fetch_html)
            if not results:
                LOG.info("Aucun thème trouvé pour l'article %s via %s", article.id, classifier.model_version)
                continue

            cxn.execute(
                text("delete from article_themes where article_id = :id and content_hash = :hash"),
                {"id": article.id, "hash": article.content_hash},
            )

            for theme, confidence in results:
                cxn.execute(
                    text(
                        """
                        insert into article_themes(
                            article_id, theme, confidence, model_version, taxonomy_version, content_hash
                        )
                        values(:article_id, :theme, :confidence, :model_version, :taxonomy_version, :content_hash)
                        """
                    ),
                    {
                        "article_id": article.id,
                        "theme": theme,
                        "confidence": float(confidence),
                        "model_version": classifier.model_version,
                        "taxonomy_version": classifier.taxonomy_version,
                        "content_hash": article.content_hash,
                    },
                )
            LOG.info("Article %s classé (%s)", article.id, classifier.model_version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classification des articles selon la source")
    parser.add_argument("--since", dest="since", help="Date (YYYY-MM-DD) à partir de laquelle relancer la classification")
    args = parser.parse_args()
    process_articles(args.since)


if __name__ == "__main__":
    main()
