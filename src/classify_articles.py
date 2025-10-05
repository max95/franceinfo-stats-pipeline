import argparse
import json
import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Iterable

import requests
from sqlalchemy import text
from dotenv import load_dotenv

from db import get_engine, init_schema

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
LOG = logging.getLogger(__name__)

CONFIG_PATH = Path("config/themes_config.json")
CACHE_PATH = Path("data/classify_cache.jsonl")
KEYWORD_MODEL_VERSION = "keyword_v1"
LLM_MODEL = os.getenv("CLASSIFY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")


def load_taxonomy() -> tuple[dict[str, list[str]], str]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    normalized = {k: [w.lower() for w in v] for k, v in data.items()}
    digest = hashlib.sha1(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    taxonomy_version = digest[:12]
    return normalized, taxonomy_version


def canonicalize_theme(name: str, taxonomy: dict[str, list[str]]) -> str | None:
    lowered = name.strip().lower()
    for theme in taxonomy.keys():
        if theme.lower() == lowered:
            return theme
    return None


def keyword_scores(text: str, taxonomy: dict[str, list[str]]) -> list[tuple[str, int]]:
    scores: list[tuple[str, int]] = []
    lowered = text.lower()
    for theme, keywords in taxonomy.items():
        count = 0
        for kw in keywords:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                count += 1
        if count:
            scores.append((theme, count))
    return scores


def keyword_to_confidence(count: int) -> float:
    base = 0.6
    if count <= 1:
        return base
    if count == 2:
        return 0.75
    if count == 3:
        return 0.85
    return 0.95


def load_cache() -> dict[str, list[tuple[str, float]]]:
    cache: dict[str, list[tuple[str, float]]] = {}
    if not CACHE_PATH.exists():
        return cache
    with CACHE_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                cache[payload["key"]] = [(item["theme"], float(item["confidence"])) for item in payload["themes"]]
            except Exception:
                continue
    return cache


def save_cache(cache: dict[str, list[tuple[str, float]]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as fh:
        for key, items in cache.items():
            fh.write(json.dumps({
                "key": key,
                "themes": [{"theme": theme, "confidence": conf} for theme, conf in items],
            }, ensure_ascii=False) + "\n")


def call_llm(text: str, taxonomy: Iterable[str]) -> list[tuple[str, float]]:
    if not OPENAI_API_KEY:
        LOG.warning("OPENAI_API_KEY manquant, impossible d'utiliser le fallback LLM")
        return []

    prompt = (
        "Tu es un classifieur d'articles d'actualité. "
        "Analyse le texte fourni et renvoie uniquement un objet JSON de la forme "
        "{\"themes\": [{\"theme\": \"nom\", \"confidence\": 0.x}]} sans aucun texte additionnel. "
        "Choisis les thèmes dans la liste suivante: " + ", ".join(sorted(taxonomy)) + ". "
        "Confidence doit être un nombre entre 0 et 1.\n\nARTICLE:\n" + text.strip()[:6000]
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant spécialisé en classification de news."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    try:
        resp = requests.post(
            f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOG.warning("Appel LLM en échec: %s", exc)
        return []

    content = resp.json()["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        LOG.warning("Réponse LLM invalide: %s", content)
        return []

    items = []
    for entry in data.get("themes", []):
        name = entry.get("theme") or entry.get("name")
        conf = entry.get("confidence")
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if not name:
            continue
        items.append((str(name), max(0.0, min(1.0, conf_val))))
    return items


def classify_text(text: str, taxonomy: dict[str, list[str]], taxonomy_version: str, cache: dict[str, list[tuple[str, float]]]) -> tuple[str, list[tuple[str, float]]]:
    scores = keyword_scores(text, taxonomy)
    if scores:
        results = [(theme, keyword_to_confidence(count)) for theme, count in scores]
        return KEYWORD_MODEL_VERSION, results

    cache_key = hashlib.sha1((taxonomy_version + text).encode("utf-8")).hexdigest()
    if cache_key in cache:
        return f"{LLM_MODEL}_cached", cache[cache_key]

    llm_results = call_llm(text, taxonomy.keys())
    if llm_results:
        cache[cache_key] = llm_results
        return LLM_MODEL, llm_results

    return KEYWORD_MODEL_VERSION, []


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
        sql = "select a.id, a.title, a.summary, a.content, a.content_hash from articles a"
        if conditions:
            sql += " where " + " and ".join(conditions)
        sql += " order by coalesce(a.published_at, a.inserted_at)"
        rows = cxn.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]


def process_articles(since: str | None = None) -> None:
    taxonomy, taxonomy_version = load_taxonomy()
    cache = load_cache()
    # Assure que le schéma minimal est en place (article_themes notamment)
    init_schema()
    eng = get_engine()
    to_classify = fetch_articles(eng, since)
    if not to_classify:
        LOG.info("Aucun article à classifier")
        return

    with eng.begin() as cxn:
        for art in to_classify:
            content_hash = art.get("content_hash")
            if not content_hash:
                LOG.debug("Article %s sans content_hash, skip", art["id"])
                continue

            cxn.execute(text("delete from article_themes where article_id = :id and content_hash != :hash"), {
                "id": art["id"],
                "hash": content_hash,
            })

            text_parts = [art.get("title"), art.get("summary"), art.get("content")]
            payload = "\n\n".join([p for p in text_parts if p])
            if not payload.strip():
                LOG.debug("Article %s sans contenu textuel", art["id"])
                continue

            model_version, results = classify_text(payload, taxonomy, taxonomy_version, cache)
            if not results:
                LOG.info("Aucun thème trouvé pour l'article %s", art["id"])
                continue

            cxn.execute(text("delete from article_themes where article_id = :id and content_hash = :hash"), {
                "id": art["id"],
                "hash": content_hash,
            })

            seen = set()
            for theme, confidence in results:
                canonical = theme if theme in taxonomy else canonicalize_theme(theme, taxonomy)
                if not canonical:
                    continue
                if canonical in seen:
                    continue
                seen.add(canonical)
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
                        "article_id": art["id"],
                        "theme": canonical,
                        "confidence": float(confidence),
                        "model_version": model_version,
                        "taxonomy_version": taxonomy_version,
                        "content_hash": content_hash,
                    },
                )
            LOG.info("Article %s classé (%s)", art["id"], model_version)

    save_cache(cache)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classification des articles selon la taxonomie locale")
    parser.add_argument("--since", dest="since", help="Date (YYYY-MM-DD) à partir de laquelle relancer la classification")
    args = parser.parse_args()
    process_articles(args.since)


if __name__ == "__main__":
    main()
