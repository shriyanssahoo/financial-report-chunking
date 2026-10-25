import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PARSED, OUT = ROOT / "data/parsed", ROOT / "data/processed"
IDX, RES = ROOT / "indexes", ROOT / "results/retrieval"
IDX.mkdir(exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

TOP_K, MAX_CHARS = 10, 2048
SKIP = {"Header", "Footer", "PageNumber"}

enc = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
bleu = BLEU(effective_order=True)


def element_chunks(elements):
    """Paper rules: Title starts a new chunk; Table is its own whole chunk;
    other elements merge until ~2048 chars without splitting an element."""
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
        pg0 = pg - 1                      # Unstructured is 1-indexed, gold pages are 0-indexed
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


def prefix_of(text, n=2):
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return " ".join(sents[:n])


def best_overlap(retrieved, golds):
    best = {"f": 0.0, "r": 0.0, "bleu": 0.0}
    for g in golds:
        for c in retrieved:
            s = rouge.score(g, c)["rougeL"]
            best["f"], best["r"] = max(best["f"], s.fmeasure), max(best["r"], s.recall)
            best["bleu"] = max(best["bleu"], bleu.sentence_score(c, [g]).score / 100)
    return best


parsed_docs = {p.stem for p in PARSED.glob("*.json")}
qs = pd.read_json(OUT / "subset_questions.jsonl", lines=True)
qs = qs[qs["doc_name"].isin(parsed_docs)]
print(f"{len(parsed_docs)} parsed docs -> {qs['doc_name'].nunique()} with questions, {len(qs)} questions")

VARIANTS = {"element_plain": lambda c: c["text"],
            "element_prefix": lambda c: prefix_of(c["text"]) + "\n" + c["text"]}

records = []
for variant, make_text in VARIANTS.items():
    for doc, dq in tqdm(qs.groupby("doc_name"), desc=variant):
        chunks = element_chunks(json.load(open(PARSED / f"{doc}.json")))
        cache = IDX / f"{doc}__{variant}.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:
            vecs = enc.encode([make_text(c) for c in chunks], batch_size=32, show_progress_bar=False)
            np.save(cache, vecs)
        for _, q in dq.iterrows():
            top = np.argsort(-(vecs @ enc.encode([q["question"]])[0]))[:TOP_K]
            ret = [chunks[i] for i in top]
            ov = best_overlap([c["text"] for c in ret], q["gold_texts"])
            records.append({
                "strategy": variant, "financebench_id": q["financebench_id"], "doc_name": doc,
                "table_like": bool(q["table_like"]),
                "page_hit": any(set(q["gold_pages"]) & set(c["pages"]) for c in ret),
                "rougeL_f": ov["f"], "rougeL_r": ov["r"], "bleu": ov["bleu"],
                "n_chunks_doc": len(chunks),
                "retrieved": [{"pages": c["pages"], "kind": c["kind"], "text": c["text"]} for c in ret],
            })

el = pd.DataFrame(records)
el.to_json(RES / "element_retrieval.jsonl", orient="records", lines=True)

base = pd.read_json(RES / "baseline_retrieval.jsonl", lines=True)
base = base[base["financebench_id"].isin(qs["financebench_id"])]   # same questions only
allr = pd.concat([base, el], ignore_index=True)

summary = allr.groupby("strategy").agg(
    questions=("page_hit", "size"),
    page_acc=("page_hit", lambda s: 100 * s.mean()),
    rougeL_f=("rougeL_f", "mean"), rougeL_r=("rougeL_r", "mean"), bleu=("bleu", "mean"),
).round(3)
print("\n", summary.to_string())
print("\nTable-like questions only (page accuracy %):")
print(allr[allr["table_like"]].groupby("strategy")["page_hit"].agg(["mean", "size"]).round(3).to_string())
summary.to_csv(RES / "comparison_summary.csv") 