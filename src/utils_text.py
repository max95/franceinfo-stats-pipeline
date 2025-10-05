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
