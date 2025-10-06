
.PHONY: init install db ingest dedupe summarize align classify dash all

.venv/.requirements-installed: requirements.txt
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	touch .venv/.requirements-installed

install: .venv/.requirements-installed

init: install
	cp -n .env.example .env || true

# Initialise le schéma DB
db: install
	. .venv/bin/activate && python src/db.py

# Ingestion RSS
ingest: install
	. .venv/bin/activate && python src/ingest_rss.py

SINCE ?=

classify: install
	. .venv/bin/activate && python src/classify_articles.py $(if $(SINCE),--since=$(SINCE),)

# Déduplication
dedupe: install
	. .venv/bin/activate && python src/dedupe.py

# Synthèses (jour + par thème)
summarize: install
	. .venv/bin/activate && python src/summarize.py

# Alignement avec tes transcriptions (assure-toi qu’elles sont en DB)
align: install
	. .venv/bin/activate && python src/align_transcripts.py

# Configuration du dashboard
DASH_START_DATE ?= 2024-01-01
DASH_END_DATE ?= 2026-01-31
DASH_PERIOD ?= W
DASH_CLUSTERS ?= 10

# Dashboard
dash: install
	. .venv/bin/activate && python -m src.descriptive_stats --start-date $(DASH_START_DATE) --end-date $(DASH_END_DATE)
	. .venv/bin/activate && python -m src.semantic_analysis --start-date $(DASH_START_DATE) --end-date $(DASH_END_DATE) --period $(DASH_PERIOD) --clusters $(DASH_CLUSTERS)
	. .venv/bin/activate && DASH_START_DATE=$(DASH_START_DATE) DASH_END_DATE=$(DASH_END_DATE) DASH_PERIOD=$(DASH_PERIOD) DASH_CLUSTERS=$(DASH_CLUSTERS) python src/dashboard_gradio.py

# Pipeline complet pour aujourd’hui
all: ingest classify summarize align
