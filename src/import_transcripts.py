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
