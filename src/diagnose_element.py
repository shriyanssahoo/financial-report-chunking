"""Why did element chunking miss some questions?

For every question where element_plain missed the gold page, check:
  - was there any element chunk on the gold page(s)?
  - how much of the gold evidence text is inside those chunks (word coverage 0..1)?
and say whether the problem is the PARSE (content lost) or the RANKING (content there, not retrieved).

Run from the repo root after run_element.py:   python src/diagnose_element.py
Optional:  python src/diagnose_element.py > diagnosis.txt     (then paste or push diagnosis.txt)
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARSED, OUT, RES = ROOT / "data/parsed", ROOT / "data/processed", ROOT / "results/retrieval"
MAX_CHARS = 2048
SKIP = {"Header", "Footer", "PageNumber"}


def element_chunks(elements):
    """Same rules as run_element.py (copied so this file can run on its own)."""
    chunks, cur, pages = [], "", set()

    def flush():
        nonlocal cur, pages
        if cur.strip():
            chunks.append({"text": cur.strip(), "pages": sorted(pages), "kind": "text"})
        cur, pages = "", set()

    for el in elements:
        cat, text = el["type"], (el.get("text") or "").strip()
        pg = el["metadata"].get("page_number")
        if not text or cat in SKIP or pg is None:
            continue
        pg0 = pg - 1
        if cat == "Table":
            flush()
            chunks.append({"text": text, "pages": [pg0], "kind": "table"})
        elif cat == "Title":
            flush()
            cur, pages = text + "\n", {pg0}
        else:
            if cur and len(cur) + len(text) > MAX_CHARS:
                flush()
            cur += text + "\n"
            pages.add(pg0)
    flush()
    return chunks


def words(t):
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2}


def main():
    qs = pd.read_json(OUT / "subset_questions.jsonl", lines=True).set_index("financebench_id")
    el = pd.read_json(RES / "element_retrieval.jsonl", lines=True)
    base = pd.read_json(RES / "baseline_retrieval.jsonl", lines=True)

    el_plain = el[el["strategy"] == "element_plain"].set_index("financebench_id")
    missed = el_plain[~el_plain["page_hit"]]
    b512 = base[base["strategy"] == "base_512"].set_index("financebench_id")["page_hit"]
    print(f"element_plain missed {len(missed)} of {len(el_plain)} questions\n")

    verdicts, cache = [], {}
    for qid, row in missed.iterrows():
        q = qs.loc[qid]
        doc = row["doc_name"]
        if doc not in cache:
            f = PARSED / f"{doc}.json"
            if not f.exists():
                print(f"[{qid}] {doc}: parsed file not found, skipped\n")
                continue
            cache[doc] = element_chunks(json.load(open(f)))
        gold = set(q["gold_pages"])
        on_gold = [c for c in cache[doc] if gold & set(c["pages"])]
        cov = 0.0
        for g in q["gold_texts"]:
            gw = words(g)
            for c in on_gold:
                if gw:
                    cov = max(cov, len(gw & words(c["text"])) / len(gw))
        if not on_gold:
            verdict = "NO CHUNKS ON GOLD PAGE (parse lost the page)"
        elif cov < 0.6:
            verdict = "CONTENT MISSING FROM PARSE (gold text not in chunks)"
        else:
            verdict = "PARSED OK, NOT RETRIEVED (ranking problem)"
        verdicts.append(verdict.split(" (")[0])
        top = row["retrieved"][0]
        print(f"[{qid}] {doc} | table_like={bool(row['table_like'])} | gold pages {sorted(gold)} "
              f"| base_512 hit: {bool(b512.get(qid, False))}")
        print(f"  Q: {q['question'][:140]}")
        print(f"  chunks on gold pages: {len(on_gold)} ({sum(c['kind'] == 'table' for c in on_gold)} tables) "
              f"| evidence coverage: {cov:.2f}")
        print(f"  VERDICT: {verdict}")
        print(f"  top-1 retrieved: page {top['pages']} [{top['kind']}] {top['text'][:110]!r}\n")

    print("SUMMARY of missed questions:")
    print(pd.Series(verdicts).value_counts().to_string() if verdicts else "  none")


if __name__ == "__main__":
    main()