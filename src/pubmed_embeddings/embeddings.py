from __future__ import annotations

import argparse
import os
import pathlib
import re
import signal
import sqlite3
import sys
import threading
from typing import Iterable

import faiss
import httpx
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


STOP_EVENT = threading.Event()


def _handle_stop_signal(_signum: int, _frame: object | None) -> None:
    STOP_EVENT.set()


def _slugify_model(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-")
    return s.lower() or "model"


def _connect_state(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_pmids (
            pmid INTEGER PRIMARY KEY
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    return conn


def _reconcile_faiss_ids_into_state(conn: sqlite3.Connection, index: faiss.Index) -> None:
    """If the FAISS file was written but state.sqlite lagged, recover PMIDs from the index."""
    if index.ntotal == 0:
        return
    ids = faiss.vector_to_array(index.id_map).astype(np.int64)
    conn.executemany(
        "INSERT OR IGNORE INTO embedded_pmids (pmid) VALUES (?)",
        [(int(x),) for x in ids],
    )
    conn.commit()


def _index_dim(index: faiss.IndexIDMap | faiss.IndexIDMap2) -> int:
    sub = faiss.downcast_index(index.index)
    return int(sub.d)


def _new_index(dim: int) -> faiss.IndexIDMap2:
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)


def _atomic_write_index(index: faiss.Index, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def _build_prompt(title: str, abstract: str) -> str:
    return f"{title.strip()} {abstract.strip()}"


def _sanitize_for_embed(text: str) -> str:
    """Remove NULs and invalid UTF-8 sequences that can cause Ollama to reject the request."""
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="replace").decode("utf-8").strip()


def _clip_embedding_text(text: str) -> str:
    """Sanitize and apply EMBEDDING_MAX_INPUT_CHARS when set (shared by Ollama and local)."""
    text = _sanitize_for_embed(text)
    env_max = (os.environ.get("EMBEDDING_MAX_INPUT_CHARS") or "").strip()
    if env_max.isdigit():
        text = text[: int(env_max)]
    return text


def _normalize_embedding_source(raw: str) -> str:
    s = raw.strip().lower()
    if s in ("tei", "tie"):
        return "local"
    return s


def _ollama_http_error_message(r: httpx.Response) -> str:
    try:
        data = r.json()
        if isinstance(data, dict) and "error" in data:
            return str(data["error"])
    except Exception:
        pass
    body = (r.text or "").strip()
    return body[:2000] if body else r.reason_phrase


def _parse_embedding_vector(data: dict) -> list[float]:
    """Ollama current API returns `embeddings` (batch); legacy returned `embedding`."""
    if "embeddings" in data:
        vecs = data["embeddings"]
        if not vecs or not isinstance(vecs, list):
            raise RuntimeError(f"Unexpected Ollama response (embeddings): {data!r}")
        first = vecs[0]
        if not isinstance(first, list):
            raise RuntimeError(f"Unexpected Ollama response (embeddings[0]): {data!r}")
        return first
    emb = data.get("embedding")
    if isinstance(emb, list):
        return emb
    raise RuntimeError(f"Unexpected Ollama response (no embeddings): {data!r}")


def _post_embed_modern(client: httpx.Client, model: str, chunk: str, truncate: bool | None) -> httpx.Response:
    """POST /api/embed — https://docs.ollama.com/api/embed"""
    body: dict = {"model": model, "input": chunk}
    if truncate is not None:
        body["truncate"] = truncate
    return client.post("/api/embed", json=body)


def _post_embed_legacy(client: httpx.Client, model: str, chunk: str) -> httpx.Response:
    return client.post("/api/embeddings", json={"model": model, "prompt": chunk})


def _request_embed_any(client: httpx.Client, model: str, chunk: str, truncate: bool | None) -> httpx.Response:
    """Try /api/embed; on 404 only, try legacy /api/embeddings."""
    r = _post_embed_modern(client, model, chunk, truncate)
    if r.status_code == 404:
        r = _post_embed_legacy(client, model, chunk)
    return r


def _response_to_vector(r: httpx.Response) -> np.ndarray:
    data = r.json()
    emb = _parse_embedding_vector(data)
    arr = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    if arr.size == 0:
        raise RuntimeError("Empty embedding from Ollama")
    faiss.normalize_L2(arr)
    return arr


def _fetch_ollama_embedding(
    client: httpx.Client,
    model: str,
    text: str,
    *,
    pmid: int | None = None,
) -> np.ndarray:
    """Call Ollama embed API; retry on 400 with alternate truncate and shorter text."""
    text = _clip_embedding_text(text)
    if not text:
        raise RuntimeError("Empty embedding input after sanitization")

    candidates: list[str] = [text]
    seen = {text}
    for lim in (262144, 131072, 65536, 32768, 16384, 8192, 4096):
        if len(text) > lim:
            c = text[:lim]
            if c not in seen:
                seen.add(c)
                candidates.append(c)

    truncate_tries: list[bool | None] = [True, None, False]

    prefix = f"[pmid {pmid}] " if pmid is not None else ""
    last_err = ""

    for ci, chunk in enumerate(candidates):
        if ci > 0:
            tqdm.write(f"{prefix}retrying with shorter input ({len(chunk)} chars) after 400")

        for trunc in truncate_tries:
            r = _request_embed_any(client, model, chunk, trunc)
            if r.is_success:
                return _response_to_vector(r)

            last_err = _ollama_http_error_message(r)
            if r.status_code == 400:
                continue
            raise RuntimeError(f"{prefix}Ollama error {r.status_code}: {last_err}")

    raise RuntimeError(f"{prefix}Ollama embed failed after retries. Last error: {last_err}")


def _eligible_articles(
    conn: sqlite3.Connection,
    limit: int | None,
) -> list[tuple[int, str, str]]:
    q = """
        SELECT pmid, title, abstract FROM articles
        WHERE title IS NOT NULL AND abstract IS NOT NULL
          AND trim(title) != '' AND trim(abstract) != ''
        ORDER BY pmid
    """
    rows = list(conn.execute(q))
    if limit is not None:
        rows = rows[:limit]
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def _embedded_set(state_conn: sqlite3.Connection) -> set[int]:
    return {int(r[0]) for r in state_conn.execute("SELECT pmid FROM embedded_pmids")}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed PubMed articles (title+abstract) via Ollama or local Sentence-Transformers into a per-model FAISS index."
    )
    p.add_argument(
        "--db",
        type=pathlib.Path,
        default=None,
        help="SQLite DB with articles table (default: <data-dir>/pubmed.sqlite)",
    )
    p.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Project data directory (default: data)",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for this model (default: <data-dir>/embeddings/<model-slug>)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name, or Hugging Face id for Sentence-Transformers when EMBEDDING_SOURCE=local; overrides EMBEDDING_MODEL",
    )
    p.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="Ollama base URL (overrides OLLAMA_BASE_URL)",
    )
    p.add_argument(
        "--local-batch-size",
        type=int,
        default=None,
        help="Articles per encode batch (overrides LOCAL_EMBED_BATCH_SIZE; local only)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Checkpoint (write FAISS + state) every N successful PMIDs (default: env EMBEDDING_CHECKPOINT_EVERY or 50)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N eligible rows (testing)",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    STOP_EVENT.clear()
    load_dotenv()
    args = parse_args(argv)

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = (args.db if args.db is not None else data_dir / "pubmed.sqlite").resolve()

    model = (args.model or os.environ.get("EMBEDDING_MODEL") or "").strip()
    if not model:
        print("Set EMBEDDING_MODEL in .env or pass --model", file=sys.stderr)
        return 2

    raw_embedding_source = (os.environ.get("EMBEDDING_SOURCE") or "ollama").strip().lower()
    if raw_embedding_source in ("tei", "tie"):
        tqdm.write(
            "Note: EMBEDDING_SOURCE=tei/tie maps to local (in-process Sentence-Transformers).",
            file=sys.stderr,
        )

    source = _normalize_embedding_source(os.environ.get("EMBEDDING_SOURCE") or "ollama")
    if source not in ("ollama", "local"):
        print(
            f"EMBEDDING_SOURCE must be ollama or local (tei/tie map to local); got {source!r}",
            file=sys.stderr,
        )
        return 2

    base_url = (args.ollama_base_url or os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip(
        "/"
    )

    lbs_env = (os.environ.get("LOCAL_EMBED_BATCH_SIZE") or "").strip()
    local_batch_size = args.local_batch_size
    if local_batch_size is None:
        if lbs_env.isdigit():
            local_batch_size = int(lbs_env)
        else:
            local_batch_size = 32
    local_batch_size = max(1, local_batch_size)

    ck_env = os.environ.get("EMBEDDING_CHECKPOINT_EVERY")
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        if ck_env:
            checkpoint_every = int(ck_env)
        else:
            checkpoint_every = 50
    checkpoint_every = max(1, checkpoint_every)

    slug = _slugify_model(model)
    out_dir = (args.out_dir if args.out_dir is not None else data_dir / "embeddings" / slug).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "vectors.faiss"
    state_path = out_dir / "state.sqlite"

    signal.signal(signal.SIGINT, _handle_stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop_signal)

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    articles_conn = sqlite3.connect(db_path)
    state_conn = _connect_state(state_path)

    eligible = _eligible_articles(articles_conn, limit=args.limit)
    embedded = _embedded_set(state_conn)

    index: faiss.IndexIDMap2 | None = None
    if index_path.exists() and index_path.stat().st_size > 0:
        idx_any = faiss.read_index(str(index_path))
        index = faiss.downcast_index(idx_any)
        if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            raise RuntimeError("Saved index must be IndexIDMap / IndexIDMap2")
        _reconcile_faiss_ids_into_state(state_conn, index)
        embedded = _embedded_set(state_conn)
        state_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("model", model),
        )
        state_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("dim", str(_index_dim(index))),
        )
        state_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("embedding_source", source),
        )
        state_conn.commit()

    pending = [(p, t, a) for p, t, a in eligible if p not in embedded]

    if not pending:
        print(f"Nothing to do: 0 pending PMIDs (eligible={len(eligible)}, embedded={len(embedded)}).")
        articles_conn.close()
        state_conn.close()
        return 0

    since_ck: list[int] = []
    processed_this_run = 0

    def checkpoint(index_obj: faiss.IndexIDMap2 | faiss.IndexIDMap) -> None:
        if index_obj.ntotal == 0 or not since_ck:
            return
        _atomic_write_index(index_obj, index_path)
        state_conn.executemany(
            "INSERT OR IGNORE INTO embedded_pmids (pmid) VALUES (?)",
            [(pmid,) for pmid in since_ck],
        )
        state_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("model", model),
        )
        state_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("dim", str(_index_dim(index_obj))),
        )
        state_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("embedding_source", source),
        )
        state_conn.commit()
        since_ck.clear()

    timeout = httpx.Timeout(600.0)

    def run_ollama_loop(client: httpx.Client) -> None:
        nonlocal index, processed_this_run
        pbar = tqdm(pending, desc=f"embed [{slug}]", unit="pmid")
        for pmid, title, abstract in pbar:
            if STOP_EVENT.is_set():
                break
            prompt = _build_prompt(title, abstract)
            try:
                vec = _fetch_ollama_embedding(client, model, prompt, pmid=pmid)
            except Exception as exc:
                tqdm.write(f"[pmid {pmid}] Ollama error: {exc}")
                raise

            dim = vec.shape[1]
            if index is None:
                index = _new_index(dim)
            else:
                if _index_dim(index) != dim:
                    raise RuntimeError(
                        f"Embedding dim mismatch: index has {_index_dim(index)}, model returned {dim}"
                    )

            ids = np.array([pmid], dtype=np.int64)
            index.add_with_ids(vec, ids)
            since_ck.append(pmid)
            processed_this_run += 1

            if len(since_ck) >= checkpoint_every:
                checkpoint(index)

    def run_local_loop() -> None:
        nonlocal index, processed_this_run
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model)
        with tqdm(total=len(pending), desc=f"embed [{slug}] local", unit="pmid") as pbar:
            i = 0
            while i < len(pending):
                if STOP_EVENT.is_set():
                    break
                batch = pending[i : i + local_batch_size]
                i += len(batch)
                pmids: list[int] = []
                inputs: list[str] = []
                for pmid, title, abstract in batch:
                    text = _clip_embedding_text(_build_prompt(title, abstract))
                    if not text:
                        raise RuntimeError(
                            f"Empty embedding input after sanitization [pmid {pmid}]"
                        )
                    pmids.append(pmid)
                    inputs.append(text)

                emb = st_model.encode(
                    inputs,
                    batch_size=min(local_batch_size, len(inputs)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                arr = np.asarray(emb, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[0] != len(batch):
                    raise RuntimeError(
                        f"local model returned {arr.shape[0]} vectors for {len(batch)} inputs"
                    )
                faiss.normalize_L2(arr)
                dim = int(arr.shape[1])

                if index is None:
                    index = _new_index(dim)
                elif _index_dim(index) != dim:
                    raise RuntimeError(
                        f"Embedding dim mismatch: index has {_index_dim(index)}, model returned {dim}"
                    )

                for j, pmid in enumerate(pmids):
                    row = arr[j : j + 1]
                    ids = np.array([pmid], dtype=np.int64)
                    index.add_with_ids(row, ids)
                    since_ck.append(pmid)
                    processed_this_run += 1

                pbar.update(len(batch))

                if index is not None and len(since_ck) >= checkpoint_every:
                    checkpoint(index)

    if source == "ollama":
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            run_ollama_loop(client)
            if index is not None and since_ck:
                checkpoint(index)
    else:
        run_local_loop()
        if index is not None and since_ck:
            checkpoint(index)

    stopped = STOP_EVENT.is_set()

    articles_conn.close()
    state_conn.close()

    if stopped:
        print(
            f"Stopped by user; progress saved up to last checkpoint "
            f"({processed_this_run} PMIDs processed this run).",
            file=sys.stderr,
        )
        return 130

    print(
        f"Done. Processed {processed_this_run} PMIDs into {out_dir} "
        f"(index={index_path.name}, state={state_path.name})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
