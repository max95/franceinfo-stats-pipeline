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
