"""Custom per-source article classifiers.

This module exposes a simple registry that can dispatch articles to the
appropriate classifier implementation based on the article metadata.  Each
classifier is responsible for scraping whatever metadata is needed for its
source and returning the list of themes that should be saved in the database.

Adding support for a new source only requires implementing a new subclass of
:class:`SourceClassifier` and appending an instance to ``REGISTRY``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from html import unescape
from typing import Callable, Iterable
from urllib.parse import urlparse

import logging

from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class Article:
    """Minimal representation of an article used during classification."""

    id: int
    source_name: str
    link: str | None
    title: str | None
    summary: str | None
    content: str | None
    content_hash: str | None
    raw: str | None


class SourceClassifier:
    """Base class for source specific classifiers."""

    #: Identifier stored in ``article_themes.model_version``
    model_version: str = "source_classifier_base"
    #: Identifier stored in ``article_themes.taxonomy_version``
    taxonomy_version: str = "source_classifier_base"

    def supports(self, article: Article) -> bool:
        """Return ``True`` if the classifier can handle the given article."""

        raise NotImplementedError

    def classify(
        self, article: Article, fetch_html: Callable[[str], str | None]
    ) -> list[tuple[str, float]]:
        """Return ``[(theme, confidence), ...]`` for the article.

        ``fetch_html`` can be used to download the raw HTML of ``article.link``
        when required by the classifier.  The base implementation returns an
        empty list so subclasses only need to override what they use.
        """

        return []


class FranceInfoClassifier(SourceClassifier):
    """Extract themes from ``<meta property="article:tag">`` entries."""

    model_version = "franceinfo_meta_tags_v1"
    taxonomy_version = "franceinfo_meta_tags_v1"

    def supports(self, article: Article) -> bool:  # type: ignore[override]
        if article.link:
            hostname = urlparse(article.link).hostname or ""
            if "franceinfo.fr" in hostname:
                return True
        return article.source_name.startswith("franceinfo")

    def classify(  # type: ignore[override]
        self, article: Article, fetch_html: Callable[[str], str | None]
    ) -> list[tuple[str, float]]:
        if not article.link:
            LOG.debug("Article %s sans lien pour France Info", article.id)
            return []

        html = fetch_html(article.link)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        tags: list[str] = []
        for meta in soup.find_all("meta", attrs={"property": "article:tag"}):
            content = meta.get("content")
            if not content:
                continue
            normalized = unescape(content).strip()
            if not normalized:
                continue
            tags.append(normalized)

        seen: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_tags.append(tag)

        return [(tag, 0.9) for tag in unique_tags]


class LactualiteClassifier(SourceClassifier):
    """Read classification categories directly from the RSS payload."""

    model_version = "lactualite_rss_categories_v1"
    taxonomy_version = "lactualite_rss_categories_v1"

    def supports(self, article: Article) -> bool:  # type: ignore[override]
        return article.source_name.startswith("lactualite")

    def classify(  # type: ignore[override]
        self, article: Article, fetch_html: Callable[[str], str | None]
    ) -> list[tuple[str, float]]:
        if not article.raw:
            LOG.debug("Article %s sans payload brut pour L'actualité", article.id)
            return []

        try:
            payload = json.loads(article.raw)
        except json.JSONDecodeError:
            LOG.warning("Article %s: JSON invalide dans raw", article.id)
            return []

        candidates: list[str] = []

        tags = payload.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict):
                    label = tag.get("term") or tag.get("label")
                    if isinstance(label, str) and label.strip():
                        candidates.append(label.strip())
                elif isinstance(tag, str) and tag.strip():
                    candidates.append(tag.strip())

        category = payload.get("category")
        if isinstance(category, str) and category.strip():
            candidates.append(category.strip())

        categories = payload.get("categories")
        if isinstance(categories, list):
            for cat in categories:
                if isinstance(cat, str) and cat.strip():
                    candidates.append(cat.strip())

        seen: set[str] = set()
        unique: list[str] = []
        for tag in candidates:
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(tag)

        return [(tag, 0.85) for tag in unique]


REGISTRY: tuple[SourceClassifier, ...] = (
    FranceInfoClassifier(),
    LactualiteClassifier(),
)


def pick_classifier(article: Article) -> SourceClassifier | None:
    """Return the first classifier that supports ``article``."""

    for classifier in REGISTRY:
        try:
            if classifier.supports(article):
                return classifier
        except Exception:  # pragma: no cover - defensive
            LOG.exception("Classifier %s failed during supports()", classifier)
    return None


def iter_supported_classifiers() -> Iterable[SourceClassifier]:
    """Expose the configured classifiers (mainly for introspection/tests)."""

    return REGISTRY
