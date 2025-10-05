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
