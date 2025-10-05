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
