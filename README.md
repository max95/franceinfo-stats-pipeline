# Pipeline RSS ↔️ Transcripts — France Info Stats

Ce document te donne **une implémentation complète** et modulaire :

1. **Ingestion RSS** (France Info, Le Monde, etc.)
2. **Stockage** (PostgreSQL ou SQLite) + schéma SQL
3. **Déduplication** multi-sources (titre + similitude)
4. **Synthèse LLM** journalière et par thème
5. **Alignement** avec tes **transcriptions France Info**
6. **Dashboard Gradio v4** (segments, résumés, stats, brouillons social)
7. **Planification** (cron / systemd timer)

Chaque bloc inclut **scripts prêts à l’emploi**, **prompts** et **commandes**.

---

## Analyse descriptive

Pour produire des statistiques descriptives (volume d’articles, poids des thèmes, durée moyenne d’un sujet, corrélations et pics médiatiques), exécute :

```
python -m src.descriptive_stats --start-date 2024-01-01 --end-date 2024-01-31 --top-themes 15
```

Passe `--top-themes 0` pour afficher tous les thèmes dans les séries temporelles.

Les bornes de dates sont optionnelles (et les paramètres `--top-themes`, `--correlate`, `--z-threshold`, `--min-days` permettent de personnaliser l'analyse). La sortie affiche :

- le nombre d’articles par jour, source et thème ;
- le poids relatif de chaque thème ;
- la durée moyenne de « vie médiatique » par thème ainsi que la moyenne générale ;
- la fréquence quotidienne des principaux thèmes ;
- une matrice de corrélation (option `--correlate énergie écologie` par exemple) ;
- la détection automatique des pics médiatiques via z-score (paramétrable avec `--z-threshold`).

---

## Analyse sémantique

Pour extraire automatiquement les mots-clés dominants, détecter les expressions émergentes et regrouper les articles similaires, assure-toi d'abord d'avoir installé les dépendances (`pip install -r requirements.txt`), puis exécute :

```
python -m src.semantic_analysis --start-date 2024-01-01 --end-date 2024-01-31 --period W --clusters 10
```

Les paramètres principaux :

- `--period` : granularité temporelle (jour `D`, semaine `W`, mois `M`, etc.) ;
- `--top-keywords` : nombre de mots-clés à conserver par période ;
- `--min-score` & `--surge-ratio` : seuils pour repérer de nouvelles expressions ou celles en forte hausse ;
- `--clusters`, `--cluster-top-terms` : configuration du regroupement automatique d’articles (K-Means sur TF-IDF).

La sortie CLI détaille :

- les mots-clés principaux pour chaque période ;
- les expressions identifiées comme nouvelles ou en forte progression ;
- la composition des clusters avec leurs termes saillants.

---

## 0) Arborescence & prérequis

```
franceinfo-stats/
├─ .env.example
├─ requirements.txt
├─ config/
│  ├─ rss_sources.yml
│  └─ prompts/
│     ├─ daily_summary.txt
│     └─ topic_summary.txt
├─ sql/
│  ├─ schema_postgres.sql
│  └─ schema_sqlite.sql
├─ data/
│  ├─ transcripts/   # tes fichiers transcript_YYYYMMDD.txt
│  └─ cache/
├─ src/
│  ├─ db.py
│  ├─ utils_text.py
│  ├─ ingest_rss.py
│  ├─ dedupe.py
│  ├─ summarize.py
│  ├─ align_transcripts.py
│  └─ dashboard_gradio.py
└─ Makefile
```

**.env.example**

```
# DB (Postgres) — sinon laisse vide pour SQLite local
DB_DSN=postgresql+psycopg2://user:pass@localhost:5432/franceinfo
# ou pour SQLite
# DB_DSN=sqlite:///franceinfo.db

# OpenAI (ou autre fournisseur compatible OpenAI API)
OPENAI_API_KEY=changeme
OPENAI_MODEL=gpt-4.1-mini
SUMMARIZE_MAX_TOKENS=800

# Divers
TIMEZONE=Europe/Paris
```

**requirements.txt**

```
python-dotenv
SQLAlchemy>=2.0
psycopg2-binary; platform_system != 'Darwin'
psycopg2; platform_system == 'Darwin'
feedparser
rapidfuzz
pandas
tqdm
gradio>=4.0.0
requests
beautifulsoup4
plotly
scikit-learn
```

**config/rss_sources.yml**

```yaml
sources:
  - name: franceinfo_titres
    url: https://www.francetvinfo.fr/titres.rss
    topic: general
  - name: franceinfo_politique
    url: https://www.francetvinfo.fr/politique.rss
    topic: politique
  - name: franceinfo_monde
    url: https://www.francetvinfo.fr/monde.rss
    topic: monde
  - name: lemonde_une
    url: https://www.lemonde.fr/rss/une.xml
    topic: general
  - name: bfmtv_all
    url: https://www.bfmtv.com/rss/info/flux-rss/flux-toutes-les-actualites/
    topic: general
```

**config/prompts/daily_summary.txt**

```
Tu es un analyste éditorial concis. Résume les faits marquants des articles (titres + descriptions) ci-dessous pour la date {date} (heure de Paris). Regroupe par thèmes explicites (Politique, Monde, Économie, Société, Science/Tech, Sport, Culture), 5 bullets max par thème. Évite les doublons. Mentionne des chiffres/dates clés si présents.
```

**config/prompts/topic_summary.txt**

```
Synthétise le thème « {topic} » pour la date {date} en 5 bullets max, en évitant redites et hypothèses. Écris au style dépouillé, neutre, factuel.
```

**sql/schema_postgres.sql**

```sql
create schema if not exists public;

create table if not exists sources (
  id serial primary key,
  name text unique not null,
  url text not null,
  topic text
);

create table if not exists articles (
  id bigserial primary key,
  source_id int references sources(id),
  guid text,
  link text not null,
  title text not null,
  summary text,
  published_at timestamptz,
  topic text,
  raw jsonb,
  inserted_at timestamptz default now(),
  unique(source_id, coalesce(guid, link))
);

-- Déduplication logique multi-sources
create table if not exists article_dupes (
  id bigserial primary key,
  article_id bigint references articles(id) on delete cascade,
  duplicate_of bigint references articles(id) on delete cascade,
  score real not null,
  reason text
);

-- Résumés
create table if not exists daily_summaries (
  id bigserial primary key,
  day date not null,
  summary_md text not null,
  created_at timestamptz default now(),
  unique(day)
);

create table if not exists topic_summaries (
  id bigserial primary key,
  day date not null,
  topic text not null,
  summary_md text not null,
  created_at timestamptz default now(),
  unique(day, topic)
);

-- Transcriptions (ingérées ailleurs dans ton pipeline)
create table if not exists transcripts (
  id bigserial primary key,
  day date not null,
  src text not null,
  chunk_range text,
  text text not null
);

-- Alignement / recoupements
create table if not exists crosslinks (
  id bigserial primary key,
  day date not null,
  article_id bigint references articles(id) on delete cascade,
  transcript_id bigint references transcripts(id) on delete cascade,
  similarity real not null,
  topic text
);
```

**sql/schema_sqlite.sql** (adapté)

```sql
create table if not exists sources (
  id integer primary key autoincrement,
  name text unique not null,
  url text not null,
  topic text
);
create table if not exists articles (
  id integer primary key autoincrement,
  source_id int,
  guid text,
  link text not null,
  title text not null,
  summary text,
  published_at text,
  topic text,
  raw text,
  inserted_at text default (datetime('now')),
  unique(source_id, COALESCE(guid, link))
);
create table if not exists article_dupes (
  id integer primary key autoincrement,
  article_id int,
  duplicate_of int,
  score real not null,
  reason text
);
create table if not exists daily_summaries (
  id integer primary key autoincrement,
  day text not null unique,
  summary_md text not null,
  created_at text default (datetime('now'))
);
create table if not exists topic_summaries (
  id integer primary key autoincrement,
  day text not null,
  topic text not null,
  summary_md text not null,
  created_at text default (datetime('now')),
  unique(day, topic)
);
create table if not exists transcripts (
  id integer primary key autoincrement,
  day text not null,
  src text not null,
  chunk_range text,
  text text not null
);
create table if not exists crosslinks (
  id integer primary key autoincrement,
  day text not null,
  article_id int,
  transcript_id int,
  similarity real not null,
  topic text
);
```

---

## 1) src/db.py (connexion unifiée + init)

```python
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DB_DSN", "sqlite:///franceinfo.db")

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DSN, future=True)
    return _engine


def init_schema():
    engine = get_engine()
    kind = "postgres" if DSN.startswith("postgresql") else "sqlite"
    schema_path = f"sql/schema_{'postgres' if kind=='postgres' else 'sqlite'}.sql"
    with engine.begin() as cxn:
        sql = open(schema_path, "r", encoding="utf-8").read()
        for stmt in [s for s in sql.split(";\n") if s.strip()]:
            cxn.execute(text(stmt))

if __name__ == "__main__":
    init_schema()
```

---

## 2) src/utils_text.py (normalisation + similarité)

```python
import re
from rapidfuzz import fuzz

_ws = re.compile(r"\s+")

def norm(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"[’'`´]", "'", s)
    s = _ws.sub(" ", s)
    return s


def sim(a: str, b: str) -> float:
    # Ratio hybride titre/desc
    return max(
        fuzz.QRatio(norm(a), norm(b)),
        fuzz.token_set_ratio(norm(a), norm(b)),
    ) / 100.0
```

---

## 3) src/ingest_rss.py (ingestion des flux)

```python
import json
import yaml
import feedparser
from datetime import datetime
from pathlib import Path
from sqlalchemy import text
from dotenv import load_dotenv
from db import get_engine

load_dotenv()
CFG = yaml.safe_load(open("config/rss_sources.yml", "r", encoding="utf-8"))

def upsert_sources():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            cxn.execute(text(
                """
                insert into sources(name, url, topic)
                values(:name, :url, :topic)
                on conflict(name) do update set url=excluded.url, topic=excluded.topic
                """
            ), s)


def ingest():
    engine = get_engine()
    with engine.begin() as cxn:
        for s in CFG["sources"]:
            feed = feedparser.parse(s["url"])  # pas de requêtes externes lourdes
            for e in feed.entries:
                published = e.get("published") or e.get("updated")
                if published:
                    try:
                        # feedparser parse au format struct_time parfois
                        if hasattr(e, "published_parsed") and e.published_parsed:
                            dt = datetime(*e.published_parsed[:6])
                        else:
                            dt = datetime.fromisoformat(published)
                    except Exception:
                        dt = None
                else:
                    dt = None
                raw = json.dumps({k: str(getattr(e, k, "")) for k in e.keys()}, ensure_ascii=False)
                cxn.execute(text(
                    """
                    insert into articles(source_id, guid, link, title, summary, published_at, topic, raw)
                    select id, :guid, :link, :title, :summary, :published_at, :topic, :raw
                    from sources where name=:src
                    on conflict (source_id, coalesce(guid, link)) do nothing
                    """
                ), {
                    "guid": getattr(e, "id", None),
                    "link": e.link,
                    "title": e.title,
                    "summary": getattr(e, "summary", None),
                    "published_at": dt,
                    "topic": s.get("topic"),
                    "raw": raw,
                    "src": s["name"],
                })

if __name__ == "__main__":
    upsert_sources()
    ingest()
```

---

## 4) src/dedupe.py (déduplication cross-sources)

```python
from sqlalchemy import text
from db import get_engine
from utils_text import sim

THRESHOLD = 0.88  # à ajuster

SQL_LOAD = """
select a.id, a.title, a.summary, a.topic
from articles a
left join article_dupes d on d.article_id = a.id
where d.id is null
order by a.published_at desc nulls last, a.id desc
limit 2000
"""

SQL_RECENT = """
select id, title, summary, topic from articles
order by published_at desc nulls last, id desc
limit 5000
"""

INSERT_DUPE = """
insert into article_dupes(article_id, duplicate_of, score, reason)
values (:aid, :bid, :score, :reason)
"""


def run():
    eng = get_engine()
    with eng.begin() as cxn:
        cand = cxn.execute(text(SQL_LOAD)).mappings().all()
        ref = cxn.execute(text(SQL_RECENT)).mappings().all()
        for a in cand:
            best = (-1.0, None)
            sA = (a["title"] or "") + " \n" + (a["summary"] or "")
            for b in ref:
                if b["id"] == a["id"]:
                    continue
                sB = (b["title"] or "") + " \n" + (b["summary"] or "")
                sc = sim(sA, sB)
                if sc > best[0]:
                    best = (sc, b["id"])
            if best[0] >= THRESHOLD:
                cxn.execute(text(INSERT_DUPE), {
                    "aid": a["id"],
                    "bid": best[1],
                    "score": float(best[0]),
                    "reason": f"title+summary sim >= {THRESHOLD}",
                })

if __name__ == "__main__":
    run()
```

---

## 4bis) src/classify_articles.py (classification par source)

La classification ne s'appuie plus sur un fichier JSON statique. Elle dispatch
désormais chaque article vers un classifieur dédié en fonction de la source
(domaine du lien ou nom du flux RSS).

- `src/classify_articles.py` orchestre l'extraction : il charge les articles
  non encore labellisés, choisit le classifieur approprié puis persiste les
  thèmes dans `article_themes`.
- `src/source_classifiers.py` contient les implémentations concrètes. Chaque
  classifieur hérite de `SourceClassifier`, expose un `model_version` et un
  `taxonomy_version` (stockés en base) et implémente `supports()` +
  `classify()`.

Exemple pour France Info : on récupère la page HTML de l'article et on lit les
balises `<meta property="article:tag">`.

Pour L'actualité (`lactualite_general`), le classifieur n'a pas besoin de
scraper la page : il lit directement les catégories exposées dans le flux RSS
(`raw`) et les enregistre comme thèmes.

Pour BFMTV, on télécharge la page de l'article puis on extrait les métadonnées
`chapitreX` / `categorieX` présentes dans le JavaScript embarqué afin d'alimenter
les thèmes.

```python
class FranceInfoClassifier(SourceClassifier):
    model_version = "franceinfo_meta_tags_v1"
    taxonomy_version = "franceinfo_meta_tags_v1"

    def supports(self, article: Article) -> bool:
        if article.link:
            hostname = urlparse(article.link).hostname or ""
            if "franceinfo.fr" in hostname:
                return True
        return article.source_name.startswith("franceinfo")

    def classify(self, article: Article, fetch_html):
        html = fetch_html(article.link)
        soup = BeautifulSoup(html, "html.parser")
        return [(meta.get("content"), 0.9) for meta in soup.find_all(...)]
```

Pour ajouter une nouvelle source, il suffit d'écrire un classifieur dédié et
de l'ajouter dans le tuple `REGISTRY`. Chaque classifieur peut implémenter la
logique HTML/API de son média sans impacter les autres.

---

## 5) src/summarize.py (synthèses quotidiennes & par thème)

- Fenêtre de recalcul : `SUMMARY_WINDOW_DAYS` (par défaut `3`) permet de recalculer
  les synthèses des *n* derniers jours. À chaque exécution, les synthèses sont
  régénérées pour cette fenêtre glissante afin d'intégrer les articles arrivés
  en retard.
- Coupe horaire : `SUMMARY_MIN_HOUR` (par défaut `9`) évite de lancer la
  synthèse du jour courant avant l'heure indiquée, limitant ainsi les mises à
  jour successives tout au long de la journée. Passer la valeur à `-1` pour
  désactiver la coupe.
- Fuseau : `SUMMARY_TIMEZONE` permet de préciser le fuseau utilisé pour
  l'évaluation de l'heure courante (sinon, le fuseau local du serveur est
  utilisé).

```python
import os
import json
from datetime import date, datetime, timedelta
from sqlalchemy import text
from db import get_engine
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except ImportError:  # pragma: no cover - fallback for very old python
    ZoneInfo = None

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TOK = int(os.getenv("SUMMARIZE_MAX_TOKENS", "800"))
WINDOW_DAYS = max(1, int(os.getenv("SUMMARY_WINDOW_DAYS", "3")))
MIN_HOUR = int(os.getenv("SUMMARY_MIN_HOUR", "9"))
TZ_NAME = os.getenv("SUMMARY_TIMEZONE")

PROMPT_DAILY = open("config/prompts/daily_summary.txt", "r", encoding="utf-8").read()
PROMPT_TOPIC = open("config/prompts/topic_summary.txt", "r", encoding="utf-8").read()

# Appel API compatible OpenAI (adaptable à autre fournisseur)
import requests
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")


def _now() -> datetime:
    """Return timezone-aware now when possible."""
    if TZ_NAME:
        if ZoneInfo is not None:
            try:
                return datetime.now(ZoneInfo(TZ_NAME))
            except Exception:
                pass
    if ZoneInfo is not None:
        try:
            return datetime.now().astimezone()
        except Exception:
            pass
    return datetime.now()


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


def iter_days(reference: date, now: datetime) -> list[date]:
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
    now = _now()
    today = now.date()
    for day in iter_days(today, now):
        build_daily(day)
        build_topics(day)
```

---

## 6) src/align_transcripts.py (recoupement RSS ↔️ transcriptions)

```python
import re
from datetime import date
from sqlalchemy import text
from db import get_engine
from utils_text import norm, sim

TOPIC_HINTS = {
    "politique": ["assemblée", "sénat", "élysée", "ministre", "élection", "parti"],
    "monde": ["ukraine", "gaza", "onu", "usa", "union européenne", "otan"],
    "économie": ["inflation", "bce", "croissance", "chômage", "pouvoir d'achat"],
}

SQL_TRANS = """
select id, day, text from transcripts where day = :day
"""
SQL_ART = """
select id, coalesce(topic,'general') as topic, title, summary
from articles
where date(coalesce(published_at, inserted_at)) = :day
and id not in (select article_id from article_dupes)
"""
INSERT = """
insert into crosslinks(day, article_id, transcript_id, similarity, topic)
values (:d, :aid, :tid, :s, :t)
"""


def guess_topic(txt: str) -> str:
    t = norm(txt)
    best = (0, "general")
    for topic, words in TOPIC_HINTS.items():
        score = sum(1 for w in words if w in t)
        if score > best[0]:
            best = (score, topic)
    return best[1]


def run(day: date):
    eng = get_engine()
    with eng.begin() as cxn:
        trs = cxn.execute(text(SQL_TRANS), {"day": str(day)}).mappings().all()
        arts = cxn.execute(text(SQL_ART), {"day": str(day)}).mappings().all()
        for tr in trs:
            ttopic = guess_topic(tr["text"]) or "general"
            for ar in arts:
                s = sim(tr["text"], ar["title"] + "\n" + (ar["summary"] or ""))
                if s >= 0.72:
                    cxn.execute(text(INSERT), {
                        "d": str(day),
                        "aid": ar["id"],
                        "tid": tr["id"],
                        "s": float(s),
                        "t": ttopic,
                    })

if __name__ == "__main__":
    from datetime import date
    run(date.today())
```

---

## 7) src/dashboard_gradio.py (v4, sans thread, avec `gr.Timer`)

```python
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
        feed = pd.read_sql_query(text(SQL_FEED), cxn.connection, params={"d": day})
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
```

---

## 8) Makefile (simplifie l’exécution locale)

```makefile
.PHONY: init db ingest dedupe summarize align dash all

init:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	cp -n .env.example .env || true

# Initialise le schéma DB
db:
	. .venv/bin/activate && python src/db.py

# Ingestion RSS
ingest:
	. .venv/bin/activate && python src/ingest_rss.py

# Déduplication
dedupe:
	. .venv/bin/activate && python src/dedupe.py

# Synthèses (jour + par thème)
summarize:
	. .venv/bin/activate && python src/summarize.py

# Alignement avec tes transcriptions (assure-toi qu’elles sont en DB)
align:
	. .venv/bin/activate && python src/align_transcripts.py

# Dashboard
dash:
	. .venv/bin/activate && python src/dashboard_gradio.py

# Pipeline complet pour aujourd’hui
all: ingest dedupe summarize align
```

---

## 9) Import de tes transcriptions existantes

Si tu as déjà des fichiers `transcript_YYYYMMDD.txt`, ajoute un script simple d’import :

```python
# src/import_transcripts.py
from pathlib import Path
from datetime import datetime
from sqlalchemy import text
from db import get_engine

p = Path("data/transcripts")
eng = get_engine()
with eng.begin() as cxn:
    for f in sorted(p.glob("transcript_*.txt")):
        day = f.stem.split("_")[-1]
        txt = f.read_text(encoding="utf-8", errors="ignore")
        cxn.execute(text("""
            insert into transcripts(day, src, chunk_range, text)
            values (:d, :s, :r, :t)
        """), {"d": f"{day[:4]}-{day[4:6]}-{day[6:]}", "s": f.name, "r": "[all]", "t": txt})
```

---

## 10) Planification (cron / systemd)

**cron** (toutes les 15 min) :

```
*/15 * * * * cd /path/franceinfo-stats && . .venv/bin/activate && make ingest dedupe summarize align >> logs/pipeline.log 2>&1
```

**systemd timer** (plus robuste) — fichiers :

`/etc/systemd/system/franceinfo.service`

```
[Unit]
Description=FranceInfo RSS Pipeline
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/path/franceinfo-stats
ExecStart=/bin/bash -lc '. .venv/bin/activate && make all'
```

`/etc/systemd/system/franceinfo.timer`

```
[Unit]
Description=Run FranceInfo pipeline every 10 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
```

Activation :

```
sudo systemctl daemon-reload
sudo systemctl enable --now franceinfo.timer
```

---

## 11) Notes de perf & coût

* **Déduplication** : ajuste `THRESHOLD` (0.85–0.92). Enregistre toujours le score pour audit.
* **Appel LLM** : limite le bundle par jour (par ex. max 200 items), sinon **échantillonne** par source/thème.
* **Cache** : tu peux mémoriser des hash (titre+lien) pour éviter les resoumissions.
* **Postgres** conseillé en prod (index sur `(date(published_at))`, `topic`).

---

## 12) Extensions rapides

* **Enrichissement plein texte** (facultatif) : récupérer le contenu HTML des liens (requests+Readability) et le stocker en `articles.full_text` pour de meilleures synthèses.
* **Exports** : `make export` pour CSV/Parquet par jour.
* **Drafts Social** : ajouter un `social_drafts` table et une fonction LLM pour tw/threads/reels (avec gabarits).
* **Comparatif biais** : page Gradio montrant les angles par source (mêmes sujets, titres différents).

---

## 13) Commandes utiles

```
make init
make db
make ingest
make dedupe
make summarize
make align
make dash
```

Ça te donne un socle propre et réutilisable. Dis-moi si tu veux que j’ajoute l’**enrichissement plein texte** + un **module “drafts social”** directement, ou si on branche ça à ta base déjà en place (SQLite ↔︎ Postgres).
