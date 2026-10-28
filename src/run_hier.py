"""Hierarchical retrieval: first find the right PAGES, then search chunks only inside those pages.

IDEA (library analogy: building -> shelf -> book)
    Level 1: every page gets one vector (a short 'digest' of the page). The question is matched
             against pages and we keep the top-P pages.
    Level 2: only chunks that touch one of those P pages are searched; the best 10 are returned.
No API / LLM calls are needed. Chunks come from either
    --chunks element : the paper's element chunking (needs data/parsed/)
    --chunks llm     : the LLM chunks already saved in indexes/llm_cache/<model>/ (no new API calls;
                       pages that were never answered use the fallback rule and are reported)

Run from the repo root:
    python src/run_hier.py --chunks element --max-pages 100
    python src/run_hier.py --chunks llm --model gemini-3.5-flash-lite --max-pages 100
Results: results/retrieval/hier_retrieval__<chunks>.jsonl and hier_comparison__<chunks>.csv
"""
import argparse
import hashlib
import json
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARSED, OUT = ROOT / "data/parsed", ROOT / "data/processed"
IDX, RES = ROOT / "indexes", ROOT / "results/retrieval"

TOP_K, MAX_CHARS, DIGEST_CHARS = 10, 2048, 150
SKIP = {"Header", "Footer", "PageNumber"}


def element_chunks(elements):
    """Same rules as run_element.py (Title starts a chunk, Table is its own chunk, ~2048 chars)."""
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


def page_digests(doc):
    """{0-indexed page: short text of the whole page}: the first ~150 chars of every element."""
    digests = {}
    for el in json.load(open(PARSED / f"{doc}.json")):
        text = (el.get("text") or "").strip().replace("\n", " ")
        pg = el["metadata"].get("page_number")
        if not text or el["type"] in SKIP or pg is None:
            continue
        digests.setdefault(pg - 1, []).append(text[:DIGEST_CHARS])
    return {pg: " | ".join(parts) for pg, parts in sorted(digests.items())}


def get_chunks(doc, mode, model):
    if mode == "element":
        return element_chunks(json.load(open(PARSED / f"{doc}.json"))), 0
    import run_llm                                     # reuse the saved LLM answers; never call the API
    run_llm.ARGS = types.SimpleNamespace(model=model, workers=1, rpm=0, provider="openai", base_url="",
                                         host="", prompt_chars=0, reasoning_effort="")
    run_llm.ABORT.set()                                # makes ask() return at once: no network calls
    chunks, n_fb, _ = run_llm.chunk_doc(doc)
    return chunks, n_fb


def hier_retrieve(qv, page_ids, page_vecs, chunks, chunk_vecs, P):
    """Level 1: top-P pages. Level 2: best TOP_K chunks that touch those pages."""
    top_pages = {page_ids[i] for i in np.argsort(-(page_vecs @ qv))[:P]}
    cand = [i for i, c in enumerate(chunks) if top_pages & set(c["pages"])]
    if not cand:                                       # safety net: fall back to the whole document
        cand = list(range(len(chunks)))
    order = np.argsort(-(chunk_vecs[cand] @ qv))[:TOP_K]
    return [cand[i] for i in order], top_pages


def best_overlap(rouge, bleu, retrieved, golds):
    best = {"f": 0.0, "r": 0.0, "bleu": 0.0}
    for g in golds:
        for c in retrieved:
            s = rouge.score(g, c)["rougeL"]
            best["f"], best["r"] = max(best["f"], s.fmeasure), max(best["r"], s.recall)
            best["bleu"] = max(best["bleu"], bleu.sentence_score(c, [g]).score / 100)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", choices=["element", "llm"], default="element")
    ap.add_argument("--model", default="gemini-3.5-flash-lite", help="for --chunks llm: which saved LLM answers")
    ap.add_argument("--pages", type=int, nargs="*", default=[3, 5, 10], help="values of P (pages kept at level 1)")
    ap.add_argument("--max-pages", type=int, default=0)
    args = ap.parse_args()

    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU
    from sentence_transformers import SentenceTransformer

    RES.mkdir(parents=True, exist_ok=True)
    IDX.mkdir(exist_ok=True)
    tag = args.chunks if args.chunks == "element" else f"llm_{re.sub(r'[^\w.-]', '_', args.model)}"

    qs = pd.read_json(OUT / "subset_questions.jsonl", lines=True)
    qs = qs[qs["doc_name"].isin({p.stem for p in PARSED.glob("*.json")})]
    el_file = RES / "element_retrieval.jsonl"
    if el_file.exists():                               # same questions as the other methods
        qs = qs[qs["financebench_id"].isin(pd.read_json(el_file, lines=True)["financebench_id"])]
    if args.max_pages:
        sd = pd.read_csv(OUT / "subset_docs.csv")
        qs = qs[qs["doc_name"].isin(set(sd.loc[sd["n_pages"] <= args.max_pages, "doc_name"]))]
    print(f"{qs['doc_name'].nunique()} docs, {len(qs)} questions | chunks: {tag} | P = {args.pages}")

    enc = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    bleu = BLEU(effective_order=True)

    def embed_cached(name, texts):
        h = hashlib.md5("\x00".join(texts).encode()).hexdigest()[:8]
        f = IDX / f"{name}__{h}.npy"
        if f.exists():
            return np.load(f)
        v = enc.encode(texts, batch_size=32, show_progress_bar=False)
        np.save(f, v)
        return v

    records = []
    for doc, dq in qs.groupby("doc_name"):
        chunks, n_fb = get_chunks(doc, args.chunks, args.model)
        digests = page_digests(doc)
        page_ids = list(digests)
        print(f"{doc}: {len(chunks)} chunks, {len(page_ids)} pages" +
              (f", {n_fb} pages used the fallback rule" if args.chunks == "llm" else ""))
        chunk_vecs = embed_cached(f"{doc}__hier_{tag}_chunks", [c["text"] for c in chunks])
        page_vecs = embed_cached(f"{doc}__hier_pages", [digests[p] for p in page_ids])
        for _, q in dq.iterrows():
            qv = enc.encode([q["question"]])[0]
            gold = set(q["gold_pages"])
            for P in args.pages:
                idx, top_pages = hier_retrieve(qv, page_ids, page_vecs, chunks, chunk_vecs, P)
                ret = [chunks[i] for i in idx]
                ov = best_overlap(rouge, bleu, [c["text"] for c in ret], q["gold_texts"])
                records.append({
                    "strategy": f"hier_{tag}_P{P}", "financebench_id": q["financebench_id"], "doc_name": doc,
                    "table_like": bool(q["table_like"]),
                    "l1_hit": bool(gold & top_pages),
                    "page_hit": any(gold & set(c["pages"]) for c in ret),
                    "rougeL_f": ov["f"], "rougeL_r": ov["r"], "bleu": ov["bleu"],
                    "n_chunks_doc": len(chunks),
                    "retrieved": [{"pages": c["pages"], "kind": c["kind"], "text": c["text"]} for c in ret],
                })

    hier = pd.DataFrame(records)
    hier.to_json(RES / f"hier_retrieval__{tag}.jsonl", orient="records", lines=True)

    parts = [hier]
    for f in sorted(RES.glob("*_retrieval*.jsonl")):
        if "_test" in f.name or f.name.startswith("hier_"):
            continue
        d = pd.read_json(f, lines=True)
        parts.append(d[d["financebench_id"].isin(qs["financebench_id"])])
    allr = pd.concat(parts, ignore_index=True)
    summary = allr.groupby("strategy").agg(
        questions=("page_hit", "size"),
        page_acc=("page_hit", lambda s: 100 * s.mean()),
        rougeL_f=("rougeL_f", "mean"), rougeL_r=("rougeL_r", "mean"), bleu=("bleu", "mean"),
    ).round(3)
    print("\n", summary.to_string())
    print("\nLevel-1 accuracy (is a gold page among the top-P pages?):")
    print(hier.groupby("strategy")["l1_hit"].mean().mul(100).round(1).to_string())
    print("\nTable-like questions only (page accuracy):")
    print(allr[allr["table_like"]].groupby("strategy")["page_hit"].agg(["mean", "size"]).round(3).to_string())
    summary.to_csv(RES / f"hier_comparison__{tag}.csv")


if __name__ == "__main__":
    main()