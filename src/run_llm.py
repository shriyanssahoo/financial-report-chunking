"""LLM-based chunking + retrieval evaluation (same protocol as run_element.py).

IDEA
    The parsed PDF gives us "elements" (Title, paragraph, Table ...). For every page we show the
    LLM the numbered elements and ask ONE thing: "which numbers start a new topic?".
    WE then cut the text at those numbers. The LLM never rewrites text, so numbers and words stay
    exactly as in the PDF and page tracking stays correct. Tables are always their own chunk.
    If the LLM's answer is unusable, that page falls back to a simple rule (new chunk at each Title).

USAGE (run from the repo root, after data/parsed/ exists)
    # 1) quick test on one document with a local Ollama model
    python src/run_llm.py --provider ollama --model qwen2.5:3b --docs ULTABEAUTY_2023Q4_EARNINGS

    # 2) full run with an online API (OpenAI-compatible: Groq, Gemini, OpenRouter, OpenAI ...)
    #    PowerShell:  $env:LLM_API_KEY="your-key"
    python src/run_llm.py --provider openai --base-url https://api.groq.com/openai/v1 \
                          --model <model-id-from-provider> --workers 4

Never put your API key in a file that you commit. Use the LLM_API_KEY environment variable.
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PARSED, OUT = ROOT / "data/parsed", ROOT / "data/processed"
IDX, RES = ROOT / "indexes", ROOT / "results/retrieval"

TOP_K, MAX_CHARS, PROMPT_CHARS = 10, 2048, 250   # PROMPT_CHARS: text of each element shown to the LLM
SKIP = {"Header", "Footer", "PageNumber"}
ARGS = None   # filled in by main()


# ----------------------------------------------------------------------------- reading elements
def load_pages(doc):
    """{0-indexed page: [ {type, text}, ... ]} in reading order, without headers/footers."""
    els = json.load(open(PARSED / f"{doc}.json"))
    pages = {}
    for el in els:
        cat, text = el["type"], (el.get("text") or "").strip()
        pg = el["metadata"].get("page_number")
        if not text or cat in SKIP or pg is None:
            continue
        pages.setdefault(pg - 1, []).append({"type": cat, "text": text})   # Unstructured is 1-indexed
    return dict(sorted(pages.items()))


# ----------------------------------------------------------------------------- talking to the LLM
def build_prompt(items):
    lines = []
    limit = getattr(ARGS, "prompt_chars", 0) or PROMPT_CHARS
    for i, it in enumerate(items, 1):
        t = it["text"].replace("\n", " ")
        if len(t) > limit:
            t = t[:limit] + " ..."
        lines.append(f"[{i}] ({it['type']}) {t}")
    return (
        "Below are the numbered elements of ONE page of a financial report "
        "(10-K, 10-Q, 8-K or earnings release).\n"
        "Group consecutive elements into chunks so that each chunk is about ONE topic and can be "
        "understood on its own (for example a heading together with the paragraphs under it). "
        "Tables are handled separately, so you only decide where text topics change.\n"
        "Reply with ONLY a JSON list of the element numbers that START a new chunk, in increasing "
        "order. Element 1 always starts a chunk.\n"
        "Example reply: [1, 4, 9]\n\n" + "\n".join(lines)
    )


def call_llm(prompt):
    msgs = [{"role": "user", "content": prompt}]
    if ARGS.provider == "ollama":
        r = requests.post(f"{ARGS.host}/api/chat", timeout=300, json={
            "model": ARGS.model, "messages": msgs, "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096}})
        r.raise_for_status()
        return r.json()["message"]["content"]
    key = os.environ.get("LLM_API_KEY", "")
    body = {"model": ARGS.model, "messages": msgs, "temperature": 0}
    if getattr(ARGS, "reasoning_effort", ""):
        body["reasoning_effort"] = ARGS.reasoning_effort      # fewer 'thinking' tokens on models that support it
    r = requests.post(f"{ARGS.base_url.rstrip('/')}/chat/completions", timeout=120,
                      headers={"Authorization": f"Bearer {key}"}, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_starts(raw, n):
    """Pull the last [..] list of integers out of the reply; keep numbers in 1..n."""
    found = re.findall(r"\[[\d\s,]*\]", raw or "")
    if not found:
        return None
    try:
        nums = json.loads(found[-1])
    except ValueError:
        return None
    nums = sorted({int(x) for x in nums if 1 <= int(x) <= n})
    return nums or None


_throttle_lock, _last_call = threading.Lock(), [0.0]
ABORT = threading.Event()
QUOTA = {"in_a_row": 0}     # counts 429 errors in a row; many in a row = daily quota used up


def preflight():
    """One tiny test call before the long run, so a bad key / model fails in seconds, not hours."""
    try:
        call_llm("Reply with exactly: [1]")
    except requests.RequestException as e:
        resp = getattr(e, "response", None)
        raise SystemExit(f"\nTest call to the LLM failed: {e}\n{resp.text[:300] if resp is not None else ''}\n"
                         "Fix the key / model / URL (is $env:LLM_API_KEY set in THIS window?) and run again.")
    print("LLM test call OK")


def throttle():
    """Keep at most --rpm calls per minute (0 = no limit). Protects against API token-per-minute limits."""
    if not ARGS.rpm:
        return
    with _throttle_lock:
        wait = _last_call[0] + 60.0 / ARGS.rpm - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def retry_wait(resp, attempt):
    """Seconds to wait after an error. For rate limits, use the time the API tells us."""
    wait = min(5 * 2 ** attempt, 60)
    if resp is not None and resp.status_code == 429:
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", resp.text)
        if m:
            wait = int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
        if wait > 180:   # that is a daily limit, not a short pause: stop instead of silently using the fallback
            raise RuntimeError(f"Rate limit needs a {wait / 60:.0f}-minute wait (daily limit?). "
                               "Re-run later: pages already answered are saved and will not be asked again.")
    return wait


def ask(items, tries=6):
    """Return a list of 'start' numbers, or None if the LLM could not give a usable answer."""
    prompt = build_prompt(items)
    for attempt in range(tries):
        if ABORT.is_set():                                   # another page already hit a fatal error
            return None
        throttle()
        try:
            starts = parse_starts(call_llm(prompt), len(items))
            QUOTA["in_a_row"] = 0
            if starts:
                return starts
        except requests.RequestException as e:
            resp = getattr(e, "response", None)
            if not getattr(ask, "warned", False):            # show the first error so problems are visible
                ask.warned = True
                body = (resp.text[:300] if resp is not None else "")
                print(f"\n!! LLM call failed: {e}\n!! {body}")
            if resp is not None and resp.status_code in (401, 403, 404):
                ABORT.set()                                  # wrong key / model: stop the whole run now
                raise RuntimeError(f"The API refused the request ({resp.status_code}). "
                                   "Check LLM_API_KEY, --model and --base-url, then run again.")
            if resp is not None and resp.status_code == 400:
                return None                                  # this page's request was bad: use the fallback
            if resp is not None and resp.status_code == 429:
                QUOTA["in_a_row"] += 1
                if QUOTA["in_a_row"] >= 6:                   # many in a row = daily quota is used up
                    ABORT.set()
                    raise RuntimeError("The API keeps saying 'quota exceeded' (6 times in a row). Your free daily "
                                       "quota is probably used up. Run the same command again later (Gemini's daily "
                                       "quota resets around 12:30 PM India time). Answered pages are saved.")
            time.sleep(retry_wait(resp, attempt))
    return None

def fallback_starts(items):
    """Simple rule used when the LLM fails: a new chunk at every Title."""
    return [1] + [i for i, it in enumerate(items, 1) if it["type"] == "Title"]


# ----------------------------------------------------------------------------- building chunks
def build_chunks(items, pg0, starts):
    starts = set(starts) | {1}
    out, cur = [], []

    def flush():
        nonlocal cur
        buf = ""
        for t in cur:                                   # safety cap: never exceed ~2048 characters
            if buf and len(buf) + len(t) > MAX_CHARS:
                out.append({"text": buf.strip(), "pages": [pg0], "kind": "text"})
                buf = ""
            buf += t + "\n"
        if buf.strip():
            out.append({"text": buf.strip(), "pages": [pg0], "kind": "text"})
        cur = []

    for i, it in enumerate(items, 1):
        if it["type"] == "Table":                       # tables are always their own chunk
            flush()
            out.append({"text": it["text"], "pages": [pg0], "kind": "table"})
        else:
            if i in starts:
                flush()
            cur.append(it["text"])
    flush()
    return out


def chunk_doc(doc):
    pages = load_pages(doc)
    cache_dir = IDX / "llm_cache" / re.sub(r"[^\w.-]", "_", ARGS.model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{doc}.json"
    cache = json.load(open(cache_file)) if cache_file.exists() else {}

    todo = [pg for pg, items in pages.items() if len(items) > 1 and str(pg) not in cache]
    save_lock = threading.Lock()

    def work(pg):
        a = ask(pages[pg])
        if a is not None:                               # only remember good answers; failures retry next run
            with save_lock:
                cache[str(pg)] = a
                cache_file.write_text(json.dumps(cache))   # saved after every page, so a stop loses nothing
        return a

    with ThreadPoolExecutor(ARGS.workers) as ex:
        list(tqdm(ex.map(work, todo), total=len(todo), desc=f"LLM {doc}"))

    chunks, n_fallback = [], 0
    for pg, items in pages.items():
        if len(items) <= 1:
            starts = [1]
        else:
            starts = cache.get(str(pg))
            if starts is None:
                starts, n_fallback = fallback_starts(items), n_fallback + 1
        chunks += build_chunks(items, pg, starts)
    return chunks, n_fallback, len(pages)


# ----------------------------------------------------------------------------- evaluation helpers
def prefix_of(text, n=2):
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return " ".join(sents[:n])


def best_overlap(rouge, bleu, retrieved, golds):
    best = {"f": 0.0, "r": 0.0, "bleu": 0.0}
    for g in golds:
        for c in retrieved:
            s = rouge.score(g, c)["rougeL"]
            best["f"], best["r"] = max(best["f"], s.fmeasure), max(best["r"], s.recall)
            best["bleu"] = max(best["bleu"], bleu.sentence_score(c, [g]).score / 100)
    return best


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama address")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")
    ap.add_argument("--workers", type=int, default=1, help="parallel LLM calls (use 2-4 for APIs)")
    ap.add_argument("--rpm", type=int, default=0,
                    help="max LLM calls per minute (0 = no limit). Free Groq: try 5")
    ap.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"],
                    help="optional: 'low' uses fewer tokens on reasoning models (e.g. gpt-oss, Gemini)")
    ap.add_argument("--prompt-chars", type=int, default=0,
                    help="characters of each element shown to the LLM (default 250; smaller = fewer tokens)")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="only documents with at most this many pages (0 = all). Extend later: cached pages are not repeated")
    ap.add_argument("--docs", nargs="*", help="only these documents (test mode, writes *_test files)")
    ARGS = ap.parse_args()

    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU
    from sentence_transformers import SentenceTransformer

    RES.mkdir(parents=True, exist_ok=True)
    IDX.mkdir(exist_ok=True)
    tag = re.sub(r"[^\w.-]", "_", ARGS.model)
    suffix = "_test" if ARGS.docs else ""

    qs = pd.read_json(OUT / "subset_questions.jsonl", lines=True)
    qs = qs[qs["doc_name"].isin({p.stem for p in PARSED.glob("*.json")})]
    el_file = RES / "element_retrieval.jsonl"
    if el_file.exists() and not ARGS.docs:               # same questions as the element run
        qs = qs[qs["financebench_id"].isin(pd.read_json(el_file, lines=True)["financebench_id"])]
    if ARGS.docs:
        qs = qs[qs["doc_name"].isin(ARGS.docs)]
    if ARGS.max_pages:
        sd = pd.read_csv(OUT / "subset_docs.csv")
        qs = qs[qs["doc_name"].isin(set(sd.loc[sd["n_pages"] <= ARGS.max_pages, "doc_name"]))]
    print(f"{qs['doc_name'].nunique()} docs, {len(qs)} questions | model {ARGS.model} ({ARGS.provider})")

    preflight()
    enc = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    bleu = BLEU(effective_order=True)
    variants = {f"llm_{tag}_plain": lambda c: c["text"],
                f"llm_{tag}_prefix": lambda c: prefix_of(c["text"]) + "\n" + c["text"]}

    records = []
    for doc, dq in qs.groupby("doc_name"):
        chunks, n_fb, n_pages = chunk_doc(doc)
        print(f"{doc}: {len(chunks)} chunks, {n_fb}/{n_pages} pages used the fallback rule")
        h = hashlib.md5("\x00".join(c["text"] for c in chunks).encode()).hexdigest()[:8]
        for variant, make_text in variants.items():
            cache = IDX / f"{doc}__{variant}__{h}.npy"
            if cache.exists():
                vecs = np.load(cache)
            else:
                vecs = enc.encode([make_text(c) for c in chunks], batch_size=32, show_progress_bar=False)
                np.save(cache, vecs)
            for _, q in dq.iterrows():
                top = np.argsort(-(vecs @ enc.encode([q["question"]])[0]))[:TOP_K]
                ret = [chunks[i] for i in top]
                ov = best_overlap(rouge, bleu, [c["text"] for c in ret], q["gold_texts"])
                records.append({
                    "strategy": variant, "financebench_id": q["financebench_id"], "doc_name": doc,
                    "table_like": bool(q["table_like"]),
                    "page_hit": any(set(q["gold_pages"]) & set(c["pages"]) for c in ret),
                    "rougeL_f": ov["f"], "rougeL_r": ov["r"], "bleu": ov["bleu"],
                    "n_chunks_doc": len(chunks), "fallback_pages": n_fb,
                    "retrieved": [{"pages": c["pages"], "kind": c["kind"], "text": c["text"]} for c in ret],
                })

    llm = pd.DataFrame(records)
    llm.to_json(RES / f"llm_retrieval__{tag}{suffix}.jsonl", orient="records", lines=True)

    parts = [llm]
    for f in ["baseline_retrieval.jsonl", "element_retrieval.jsonl"]:
        if (RES / f).exists():
            d = pd.read_json(RES / f, lines=True)
            parts.append(d[d["financebench_id"].isin(qs["financebench_id"])])   # same questions only
    allr = pd.concat(parts, ignore_index=True)
    summary = allr.groupby("strategy").agg(
        questions=("page_hit", "size"),
        page_acc=("page_hit", lambda s: 100 * s.mean()),
        rougeL_f=("rougeL_f", "mean"), rougeL_r=("rougeL_r", "mean"), bleu=("bleu", "mean"),
    ).round(3)
    print("\n", summary.to_string())
    print("\nTable-like questions only (page accuracy):")
    print(allr[allr["table_like"]].groupby("strategy")["page_hit"].agg(["mean", "size"]).round(3).to_string())
    summary.to_csv(RES / f"llm_comparison__{tag}{suffix}.csv")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:                 # fatal API problems: show one clear line, not a traceback
        raise SystemExit(f"\nSTOPPED: {e}")