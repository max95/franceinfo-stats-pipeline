
.PHONY: init install db ingest dedupe summarize align dash all

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

# Déduplication
dedupe: install
	. .venv/bin/activate && python src/dedupe.py

# Synthèses (jour + par thème)
summarize: install
	. .venv/bin/activate && python src/summarize.py

# Alignement avec tes transcriptions (assure-toi qu’elles sont en DB)
align: install
	. .venv/bin/activate && python src/align_transcripts.py

# Dashboard
dash: install
	. .venv/bin/activate && python src/dashboard_gradio.py

# Pipeline complet pour aujourd’hui
all: ingest dedupe summarize align
