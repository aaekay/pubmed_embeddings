from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
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


def _resolve_local_sentence_transformer_model(name: str) -> str:
    """Map shorthand or wrong-org ids to valid Hugging Face repo ids for SentenceTransformer."""
    key = name.strip()
    if not key:
        return key
    aliases = {
        "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
        "sentence-transformers/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    }
    return aliases.get(key, aliases.get(key.lower(), key))


def _resolve_local_embed_device(cli_override: str | None) -> str:
    """Pick torch device for local SentenceTransformer: cuda, cpu, or mps."""
    import torch

    raw = (cli_override or os.environ.get("LOCAL_EMBED_DEVICE") or "auto").strip().lower()
    if raw in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LOCAL_EMBED_DEVICE=cuda (or --local-device cuda) but CUDA is not available. "
                "Use a PyTorch build that matches your NVIDIA driver, update the driver, "
                "or set LOCAL_EMBED_DEVICE=cpu."
            )
        return "cuda"
    if raw == "cpu":
        return "cpu"
    if raw == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError(
                "LOCAL_EMBED_DEVICE=mps but MPS is not available (requires macOS Apple Silicon)."
            )
        return "mps"
    raise ValueError(
        "LOCAL_EMBED_DEVICE / --local-device must be auto, cuda, cpu, or mps; "
        f"got {raw!r}"
    )


def _resolve_local_batch_size(cli_override: int | None) -> tuple[int, str]:
    """Batch size for local encode: CLI > LOCAL_EMBED_BATCH_SIZE > TEI_BATCH_SIZE > default 32."""
    if cli_override is not None:
        return max(1, cli_override), "--local-batch-size"
    lbs = (os.environ.get("LOCAL_EMBED_BATCH_SIZE") or "").strip()
    if lbs.isdigit():
        return max(1, int(lbs)), "LOCAL_EMBED_BATCH_SIZE"
    tei = (os.environ.get("TEI_BATCH_SIZE") or "").strip()
    if tei.isdigit():
        return max(1, int(tei)), "TEI_BATCH_SIZE"
    return 32, "default (32)"


def _announce_local_embed_device(cli_override: str | None) -> str:
    """Resolve device and print which accelerator local embeddings will use (stderr)."""
    import torch

    embed_device = _resolve_local_embed_device(cli_override)
    tqdm.write(f"Local embeddings device: {embed_device}", file=sys.stderr)
    if embed_device == "cuda" and torch.cuda.is_available():
        n = torch.cuda.device_count()
        name0 = torch.cuda.get_device_name(0)
        if n > 1:
            tqdm.write(f"  GPU (default cuda:0 of {n}): {name0}", file=sys.stderr)
        else:
            tqdm.write(f"  GPU: {name0}", file=sys.stderr)
    elif embed_device == "mps":
        tqdm.write("  Apple Silicon GPU (MPS)", file=sys.stderr)
    return embed_device


def _connect_state(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
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
    return raw.strip().lower()


def _parse_embedding_source(raw: str) -> str:
    source = _normalize_embedding_source(raw)
    if source in ("tei", "tie"):
        raise ValueError(
            "EMBEDDING_SOURCE=tei/tie is ambiguous. "
            "Use EMBEDDING_SOURCE=local for in-process Sentence-Transformers "
            "or EMBEDDING_SOURCE=tei-http for TEI servers."
        )
    return source


def _parse_tei_http_base_urls() -> list[str]:
    """Comma-separated TEI_BASE_URLS, or single TEI_BASE_URL (no trailing slash)."""
    raw = (os.environ.get("TEI_BASE_URLS") or "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    one = (os.environ.get("TEI_BASE_URL") or "").strip()
    if one:
        return [one.rstrip("/")]
    return []


def _tei_http_embed_path() -> str:
    p = (os.environ.get("TEI_EMBED_PATH") or "/embed").strip()
    return p if p.startswith("/") else f"/{p}"


def _tei_http_headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    tok = (os.environ.get("TEI_API_KEY") or os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _tei_response_to_matrix(data: object, *, expected_rows: int) -> np.ndarray:
    """Parse TEI /embed JSON into float32 (expected_rows, dim); L2-normalize after."""
    if isinstance(data, dict):
        if "embeddings" in data and isinstance(data["embeddings"], list):
            data = data["embeddings"]
        else:
            raise RuntimeError(f"Unexpected TEI JSON object (expected list or embeddings key): {data!r}")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected TEI response (empty or not a list): {data!r}")
    first = data[0]
    if isinstance(first, (int, float)):
        if expected_rows != 1:
            raise RuntimeError(
                f"TEI returned one flat vector but batch size was {expected_rows}"
            )
        arr = np.asarray(data, dtype=np.float32).reshape(1, -1)
    else:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected TEI embedding shape: {arr.shape}")
        if arr.shape[0] != expected_rows:
            raise RuntimeError(
                f"TEI returned {arr.shape[0]} rows, expected {expected_rows}"
            )
    if arr.size == 0:
        raise RuntimeError("Empty embedding matrix from TEI")
    faiss.normalize_L2(arr)
    return arr


def _tei_truncate_default() -> bool:
    v = (os.environ.get("TEI_TRUNCATE") or "true").strip().lower()
    return v not in ("0", "false", "no")


def _post_tei_embed(
    client: httpx.Client,
    base_url: str,
    inputs: list[str],
    *,
    truncate: bool | None = None,
) -> np.ndarray:
    """POST TEI /embed; returns L2-normalized float32 matrix (len(inputs), dim)."""
    if truncate is None:
        truncate = _tei_truncate_default()
    path = _tei_http_embed_path()
    url = f"{base_url.rstrip('/')}{path}"
    body: dict = {"inputs": inputs, "truncate": truncate}
    r = client.post(url, json=body)
    if not r.is_success:
        snippet = (r.text or "")[:2000]
        raise RuntimeError(f"TEI HTTP {r.status_code} at {url}: {snippet}")
    return _tei_response_to_matrix(r.json(), expected_rows=len(inputs))


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


def _attach_state_database(
    conn: sqlite3.Connection,
    *,
    alias: str,
    path: pathlib.Path,
) -> None:
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(path),))


def _prepare_pending_articles_connection(
    db_path: pathlib.Path,
    *,
    state_path: pathlib.Path,
    canonical_state_path: pathlib.Path | None,
) -> tuple[sqlite3.Connection, bool]:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    _attach_state_database(conn, alias="shard_state", path=state_path)
    include_canonical_state = canonical_state_path is not None and canonical_state_path.exists()
    if include_canonical_state and canonical_state_path is not None:
        _attach_state_database(conn, alias="canonical_state", path=canonical_state_path)
    return conn, include_canonical_state


def _build_pending_articles_query(
    *,
    limit: int | None,
    shard_id: int | None,
    num_shards: int | None,
    last_pmid: int | None,
    batch_size: int | None,
    include_canonical_state: bool,
    count_only: bool,
) -> tuple[str, list[int]]:
    params: list[int] = []
    if limit is None:
        if count_only:
            lines = ["SELECT COUNT(*) FROM articles e"]
        else:
            lines = ["SELECT e.pmid, e.title, e.abstract FROM articles e"]
        lines.extend(
            [
                "WHERE e.title IS NOT NULL AND e.abstract IS NOT NULL",
                "  AND trim(e.title) != '' AND trim(e.abstract) != ''",
            ]
        )
    else:
        lines = [
            "WITH eligible_base AS (",
            "    SELECT pmid, title, abstract FROM articles",
            "    WHERE title IS NOT NULL AND abstract IS NOT NULL",
            "      AND trim(title) != '' AND trim(abstract) != ''",
            "    ORDER BY pmid",
            "    LIMIT ?",
            ")",
        ]
        params.append(limit)
        if count_only:
            lines.append("SELECT COUNT(*) FROM eligible_base e")
        else:
            lines.append("SELECT e.pmid, e.title, e.abstract FROM eligible_base e")
        lines.append("WHERE 1=1")
    if last_pmid is not None:
        lines.append("  AND e.pmid > ?")
        params.append(last_pmid)
    if shard_id is not None and num_shards is not None:
        lines.append("  AND (e.pmid % ?) = ?")
        params.extend([num_shards, shard_id])
    lines.append(
        "  AND NOT EXISTS (SELECT 1 FROM shard_state.embedded_pmids s WHERE s.pmid = e.pmid)"
    )
    if include_canonical_state:
        lines.append(
            "  AND NOT EXISTS (SELECT 1 FROM canonical_state.embedded_pmids c WHERE c.pmid = e.pmid)"
        )
    if not count_only:
        lines.append("ORDER BY e.pmid")
        if batch_size is not None:
            lines.append("LIMIT ?")
            params.append(batch_size)
    return "\n".join(lines), params


def _count_pending_articles(
    conn: sqlite3.Connection,
    *,
    limit: int | None,
    shard_id: int | None,
    num_shards: int | None,
    include_canonical_state: bool,
) -> int:
    query, params = _build_pending_articles_query(
        limit=limit,
        shard_id=shard_id,
        num_shards=num_shards,
        last_pmid=None,
        batch_size=None,
        include_canonical_state=include_canonical_state,
        count_only=True,
    )
    row = conn.execute(query, params).fetchone()
    return int(row[0]) if row is not None else 0


def _fetch_pending_article_batch(
    conn: sqlite3.Connection,
    *,
    limit: int | None,
    shard_id: int | None,
    num_shards: int | None,
    last_pmid: int | None,
    batch_size: int,
    include_canonical_state: bool,
) -> list[tuple[int, str, str]]:
    query, params = _build_pending_articles_query(
        limit=limit,
        shard_id=shard_id,
        num_shards=num_shards,
        last_pmid=last_pmid,
        batch_size=max(1, batch_size),
        include_canonical_state=include_canonical_state,
        count_only=False,
    )
    rows = conn.execute(query, params).fetchall()
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def _parse_gpu_ids(raw: str | None, workers: int) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(range(workers))
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    ids = [int(p) for p in parts]
    if len(ids) != workers:
        raise ValueError(
            f"--gpu-ids must list exactly {workers} device id(s), got {len(ids)}: {raw!r}"
        )
    return ids


def _merge_embedding_shards(out_dir: pathlib.Path, model: str, source: str) -> int:
    """Merge out_dir/shards/*/vectors.faiss into out_dir/vectors.faiss and union state DBs."""
    shards_root = out_dir / "shards"
    if not shards_root.is_dir():
        print(f"No shards directory: {shards_root}", file=sys.stderr)
        return 1

    shard_dirs: list[tuple[int, pathlib.Path]] = []
    for p in sorted(shards_root.iterdir()):
        if not p.is_dir():
            continue
        try:
            sid = int(p.name)
        except ValueError:
            continue
        vf = p / "vectors.faiss"
        if vf.exists() and vf.stat().st_size > 0:
            shard_dirs.append((sid, p.resolve()))

    if not shard_dirs:
        print(f"No non-empty shard indices under {shards_root}", file=sys.stderr)
        return 1

    shard_dirs.sort(key=lambda x: x[0])
    # Use IndexIDMap as merge target; merge_from supports IDMap + IDMap2 mixes safely.
    merged: faiss.IndexIDMap | None = None
    dim_ref: int | None = None

    for sid, shard_path in shard_dirs:
        idx_any = faiss.read_index(str(shard_path / "vectors.faiss"))
        idx = faiss.downcast_index(idx_any)
        if not isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            print(f"Shard {sid}: expected IndexIDMap / IndexIDMap2", file=sys.stderr)
            return 1
        d = _index_dim(idx)
        if idx.ntotal == 0:
            continue
        if dim_ref is None:
            dim_ref = d
        elif d != dim_ref:
            print(
                f"Shard {sid}: dimension {d} != {dim_ref} from previous shards",
                file=sys.stderr,
            )
            return 1
        if merged is None:
            merged = faiss.IndexIDMap(faiss.IndexFlatIP(d))
        merged.merge_from(idx)

    if merged is None or merged.ntotal == 0:
        print("Nothing to merge: all shard indices are empty.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_index(merged, out_dir / "vectors.faiss")

    state_out = out_dir / "state.sqlite"
    if state_out.exists():
        state_out.unlink()
    state_conn = _connect_state(state_out)
    seen: set[int] = set()
    for sid, shard_path in shard_dirs:
        sp = shard_path / "state.sqlite"
        if not sp.exists():
            continue
        other = sqlite3.connect(sp)
        for (pmid,) in other.execute("SELECT pmid FROM embedded_pmids"):
            pid = int(pmid)
            if pid in seen:
                print(
                    f"Duplicate PMID {pid} in shard states (merge conflict).",
                    file=sys.stderr,
                )
                other.close()
                state_conn.close()
                return 1
            seen.add(pid)
        other.close()

    state_conn.executemany(
        "INSERT OR IGNORE INTO embedded_pmids (pmid) VALUES (?)",
        [(p,) for p in sorted(seen)],
    )
    state_conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("model", model),
    )
    state_conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("dim", str(_index_dim(merged))),
    )
    state_conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("embedding_source", source),
    )
    state_conn.commit()
    state_conn.close()

    tqdm.write(
        f"Merged {len(shard_dirs)} shard(s) -> {out_dir / 'vectors.faiss'} "
        f"({merged.ntotal} vectors, {len(seen)} PMIDs in state).",
        file=sys.stderr,
    )
    return 0


def _build_worker_argv(args: argparse.Namespace, shard_id: int, num_shards: int) -> list[str]:
    """Argv for a worker subprocess (same flags, forced single-worker + shard)."""
    parts: list[str] = []
    if args.db is not None:
        parts += ["--db", str(args.db)]
    parts += ["--data-dir", str(args.data_dir)]
    if args.out_dir is not None:
        parts += ["--out-dir", str(args.out_dir)]
    if args.model is not None:
        parts += ["--model", args.model]
    if args.ollama_base_url is not None:
        parts += ["--ollama-base-url", args.ollama_base_url]
    if args.local_batch_size is not None:
        parts += ["--local-batch-size", str(args.local_batch_size)]
    if args.local_device is not None:
        parts += ["--local-device", args.local_device]
    if args.checkpoint_every is not None:
        parts += ["--checkpoint-every", str(args.checkpoint_every)]
    if args.limit is not None:
        parts += ["--limit", str(args.limit)]
    parts += [
        "--workers",
        "1",
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
    ]
    return parts


def _run_coordinator(
    args: argparse.Namespace,
    workers: int,
    gpu_ids: list[int],
    *,
    source: str,
) -> int:
    """Spawn one embedding subprocess per shard.

    For ``local``, sets ``CUDA_VISIBLE_DEVICES`` per worker. For ``tei-http``, workers
    are CPU-only HTTP clients; TEI processes should be bound to GPUs separately.
    """
    if source == "local":
        import torch

        if not torch.cuda.is_available():
            print(
                "Multi-GPU local embedding requires CUDA (torch.cuda.is_available()).",
                file=sys.stderr,
            )
            return 1
        if workers > int(torch.cuda.device_count()):
            print(
                f"--workers={workers} exceeds visible CUDA devices ({torch.cuda.device_count()}).",
                file=sys.stderr,
            )
            return 1
    elif source == "tei-http":
        urls = _parse_tei_http_base_urls()
        if not urls:
            print(
                "EMBEDDING_SOURCE=tei-http with --workers > 1 requires TEI_BASE_URL or TEI_BASE_URLS.",
                file=sys.stderr,
            )
            return 2
        if workers > len(urls):
            tqdm.write(
                f"Note: {workers} workers but only {len(urls)} TEI base URL(s); "
                f"shards map with modulo (same URL may serve multiple shards).",
                file=sys.stderr,
            )
    else:
        print(f"_run_coordinator: unsupported source {source!r}", file=sys.stderr)
        return 2

    children: list[subprocess.Popen] = []

    def _kill_children() -> None:
        for c in children:
            if c.poll() is None:
                c.terminate()
                try:
                    c.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    c.kill()

    def _coordinator_sig(signum: int, frame: object | None) -> None:
        _kill_children()
        STOP_EVENT.set()

    old_int = signal.signal(signal.SIGINT, _coordinator_sig)
    old_term = None
    if hasattr(signal, "SIGTERM"):
        old_term = signal.signal(signal.SIGTERM, _coordinator_sig)

    try:
        exe = sys.executable
        mod = "pubmed_embeddings.embeddings"
        for k in range(workers):
            env = os.environ.copy()
            if source == "local":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[k])
                tqdm.write(
                    f"Starting worker shard {k}/{workers} on GPU {gpu_ids[k]} …",
                    file=sys.stderr,
                )
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)
                ulist = _parse_tei_http_base_urls()
                u = ulist[k % len(ulist)] if ulist else "?"
                tqdm.write(
                    f"Starting tei-http worker shard {k}/{workers} (TEI {u}) …",
                    file=sys.stderr,
                )
            wargv = [exe, "-m", mod] + _build_worker_argv(args, k, workers)
            children.append(
                subprocess.Popen(
                    wargv,
                    env=env,
                    cwd=os.getcwd(),
                )
            )
        codes: list[int] = []
        for c in children:
            codes.append(c.wait())
    finally:
        signal.signal(signal.SIGINT, old_int)
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)

    bad = [i for i, code in enumerate(codes) if code != 0]
    if bad:
        tqdm.write(
            f"Worker(s) {bad} exited with codes {[codes[i] for i in bad]}.",
            file=sys.stderr,
        )
        return codes[bad[0]] if codes[bad[0]] else 1

    if getattr(args, "no_auto_merge", False):
        tqdm.write(
            f"All workers finished. Merge with: uv run pubmed-embed --merge-shards "
            f"--data-dir {args.data_dir} --model {args.model or os.environ.get('EMBEDDING_MODEL', '')} …",
            file=sys.stderr,
        )
        return 0

    model = (args.model or os.environ.get("EMBEDDING_MODEL") or "").strip()
    slug = _slugify_model(model)
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else args.data_dir.resolve() / "embeddings" / slug
    ).resolve()
    source = _normalize_embedding_source(os.environ.get("EMBEDDING_SOURCE") or "ollama")
    return _merge_embedding_shards(out_dir, model, source)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed PubMed articles (title+abstract) via Ollama, local Sentence-Transformers, or TEI HTTP into a per-model FAISS index."
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
        help="Ollama model name; Hugging Face id for local ST; metadata/path slug for tei-http; overrides EMBEDDING_MODEL",
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
        "--local-device",
        type=str,
        default=None,
        help="Torch device for local embeddings: auto, cuda, cpu, mps (overrides LOCAL_EMBED_DEVICE)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Checkpoint (write FAISS + state) every N successful PMIDs (default: env EMBEDDING_CHECKPOINT_EVERY, else 5000 local / 50 Ollama)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N eligible rows (testing)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers: local=one GPU-bound process per worker; tei-http=one HTTP client per worker (set TEI_BASE_URLS). Default: env EMBED_NUM_WORKERS or 1.",
    )
    p.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids for workers (default: 0,1,...,workers-1). Must match --workers count.",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Internal: embed only PMIDs with pmid %% num_shards == shard-id; output under <out>/shards/<shard-id>/",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Internal: total shard count (used with --shard-id).",
    )
    p.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge out-dir/shards/*/ into canonical vectors.faiss + state.sqlite (local or tei-http) and exit.",
    )
    p.add_argument(
        "--no-auto-merge",
        action="store_true",
        help="After parallel workers finish, do not merge shards automatically.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    STOP_EVENT.clear()
    load_dotenv()
    args = parse_args(argv)

    workers = args.workers
    if workers is None:
        wenv = (os.environ.get("EMBED_NUM_WORKERS") or "").strip()
        workers = int(wenv) if wenv.isdigit() else 1
    workers = max(1, int(workers))

    shard_id = args.shard_id
    num_shards = args.num_shards

    if shard_id is not None or num_shards is not None:
        if shard_id is None or num_shards is None:
            print("Both --shard-id and --num-shards are required together.", file=sys.stderr)
            return 2
        if num_shards < 1 or shard_id < 0 or shard_id >= num_shards:
            print(
                f"Invalid shard: --shard-id {shard_id} with --num-shards {num_shards}",
                file=sys.stderr,
            )
            return 2
        if workers > 1:
            print(
                "Use --workers 1 when passing --shard-id / --num-shards (coordinator sets workers).",
                file=sys.stderr,
            )
            return 2

    if args.merge_shards and workers > 1:
        print("Use --merge-shards without --workers > 1 (unset EMBED_NUM_WORKERS or use --workers 1).", file=sys.stderr)
        return 2

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = (args.db if args.db is not None else data_dir / "pubmed.sqlite").resolve()

    model = (args.model or os.environ.get("EMBEDDING_MODEL") or "").strip()
    if not model:
        print("Set EMBEDDING_MODEL in .env or pass --model", file=sys.stderr)
        return 2

    raw_embedding_source = os.environ.get("EMBEDDING_SOURCE") or "ollama"
    try:
        source = _parse_embedding_source(raw_embedding_source)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if source not in ("ollama", "local", "tei-http"):
        print(
            f"EMBEDDING_SOURCE must be ollama, local, or tei-http; got {source!r}",
            file=sys.stderr,
        )
        return 2

    slug = _slugify_model(model)
    base_out_dir = (args.out_dir if args.out_dir is not None else data_dir / "embeddings" / slug).resolve()

    if args.merge_shards:
        if source not in ("local", "tei-http"):
            print(
                "--merge-shards requires EMBEDDING_SOURCE=local or tei-http.",
                file=sys.stderr,
            )
            return 2
        return _merge_embedding_shards(base_out_dir, model, source)

    if workers > 1:
        if source not in ("local", "tei-http"):
            print(
                "--workers > 1 requires EMBEDDING_SOURCE=local or tei-http.",
                file=sys.stderr,
            )
            return 2
        if source == "local":
            try:
                gpu_ids = _parse_gpu_ids(args.gpu_ids, workers)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            return _run_coordinator(args, workers, gpu_ids, source="local")
        return _run_coordinator(args, workers, [], source="tei-http")

    local_embed_device: str | None = None
    embed_batch_size, embed_batch_from = _resolve_local_batch_size(args.local_batch_size)
    if source == "local":
        local_embed_device = _announce_local_embed_device(args.local_device)
        tqdm.write(
            f"Local embeddings batch size: {embed_batch_size} (from {embed_batch_from})",
            file=sys.stderr,
        )
    elif source == "tei-http":
        if not _parse_tei_http_base_urls():
            print(
                "Set TEI_BASE_URL or TEI_BASE_URLS for EMBEDDING_SOURCE=tei-http.",
                file=sys.stderr,
            )
            return 2
        tqdm.write(
            f"TEI HTTP batch size: {embed_batch_size} (from {embed_batch_from})",
            file=sys.stderr,
        )

    base_url = (args.ollama_base_url or os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip(
        "/"
    )

    ck_env = os.environ.get("EMBEDDING_CHECKPOINT_EVERY")
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        if ck_env:
            checkpoint_every = int(ck_env)
        else:
            # Local / TEI-HTTP runs rewrite the full FAISS file each checkpoint; a larger default avoids disk-bound slowdowns at scale.
            checkpoint_every = 5000 if source in ("local", "tei-http") else 50
    checkpoint_every = max(1, checkpoint_every)
    if source in ("local", "tei-http"):
        tqdm.write(
            f"Checkpoint every {checkpoint_every} PMIDs (FAISS + state); "
            f"set EMBEDDING_CHECKPOINT_EVERY or --checkpoint-every to tune.",
            file=sys.stderr,
        )

    if shard_id is not None:
        assert num_shards is not None
        out_dir = (base_out_dir / "shards" / str(shard_id)).resolve()
    else:
        out_dir = base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "vectors.faiss"
    state_path = out_dir / "state.sqlite"

    signal.signal(signal.SIGINT, _handle_stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop_signal)

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    state_conn = _connect_state(state_path)
    canonical_state_path = (base_out_dir / "state.sqlite") if shard_id is not None else None

    index: faiss.IndexIDMap2 | None = None
    if index_path.exists() and index_path.stat().st_size > 0:
        idx_any = faiss.read_index(str(index_path))
        index = faiss.downcast_index(idx_any)
        if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            raise RuntimeError("Saved index must be IndexIDMap / IndexIDMap2")
        _reconcile_faiss_ids_into_state(state_conn, index)
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

    articles_conn, include_canonical_state = _prepare_pending_articles_connection(
        db_path,
        state_path=state_path,
        canonical_state_path=canonical_state_path,
    )
    pending_count = _count_pending_articles(
        articles_conn,
        limit=args.limit,
        shard_id=shard_id,
        num_shards=num_shards,
        include_canonical_state=include_canonical_state,
    )

    if pending_count == 0:
        print("Nothing to do: 0 pending PMIDs.")
        articles_conn.close()
        state_conn.close()
        return 0

    pbar_desc = (
        f"embed [{slug}] {source} s{shard_id}/{num_shards}"
        if shard_id is not None
        else f"embed [{slug}] {source}"
    )

    since_ck: list[int] = []
    processed_this_run = 0

    def checkpoint(index_obj: faiss.IndexIDMap2 | faiss.IndexIDMap) -> None:
        if index_obj.ntotal == 0 or not since_ck:
            return
        t0 = time.monotonic()
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
        dt = time.monotonic() - t0
        tqdm.write(
            f"Checkpoint: ntotal={index_obj.ntotal} vectors, {dt:.2f}s (FAISS write + SQLite)",
            file=sys.stderr,
        )

    timeout = httpx.Timeout(600.0)

    def run_ollama_loop(client: httpx.Client) -> None:
        nonlocal index, processed_this_run
        last_pmid: int | None = None
        with tqdm(total=pending_count, desc=pbar_desc, unit="pmid") as pbar:
            while not STOP_EVENT.is_set():
                batch = _fetch_pending_article_batch(
                    articles_conn,
                    limit=args.limit,
                    shard_id=shard_id,
                    num_shards=num_shards,
                    last_pmid=last_pmid,
                    batch_size=1,
                    include_canonical_state=include_canonical_state,
                )
                if not batch:
                    break
                pmid, title, abstract = batch[0]
                last_pmid = pmid
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
                pbar.update(1)

                if len(since_ck) >= checkpoint_every:
                    checkpoint(index)

    def run_local_loop() -> None:
        nonlocal index, processed_this_run
        from sentence_transformers import SentenceTransformer

        assert local_embed_device is not None
        hf_model_id = _resolve_local_sentence_transformer_model(model)
        st_model = SentenceTransformer(hf_model_id, device=local_embed_device)
        use_fp16 = (os.environ.get("LOCAL_EMBED_FP16") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if use_fp16 and local_embed_device == "cuda":
            st_model.half()
            tqdm.write(
                "LOCAL_EMBED_FP16=1: model in fp16; vectors cast to float32 for FAISS (validate vs float32 if needed).",
                file=sys.stderr,
            )
        elif use_fp16:
            tqdm.write(
                "LOCAL_EMBED_FP16 ignored (use with LOCAL_EMBED_DEVICE=cuda).",
                file=sys.stderr,
            )

        last_pmid: int | None = None
        with tqdm(total=pending_count, desc=pbar_desc, unit="pmid") as pbar:
            while not STOP_EVENT.is_set():
                batch = _fetch_pending_article_batch(
                    articles_conn,
                    limit=args.limit,
                    shard_id=shard_id,
                    num_shards=num_shards,
                    last_pmid=last_pmid,
                    batch_size=embed_batch_size,
                    include_canonical_state=include_canonical_state,
                )
                if not batch:
                    break
                last_pmid = batch[-1][0]
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
                    batch_size=min(embed_batch_size, len(inputs)),
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

                ids_np = np.asarray(pmids, dtype=np.int64)
                index.add_with_ids(arr, ids_np)
                since_ck.extend(pmids)
                processed_this_run += len(pmids)

                pbar.update(len(batch))

                if index is not None and len(since_ck) >= checkpoint_every:
                    checkpoint(index)

    def run_tei_http_loop() -> None:
        nonlocal index, processed_this_run
        tei_all = _parse_tei_http_base_urls()
        if not tei_all:
            raise RuntimeError("tei-http requires TEI_BASE_URL or TEI_BASE_URLS")
        if shard_id is not None and num_shards is not None:
            base_pick = tei_all[shard_id % len(tei_all)]
            tqdm.write(f"TEI HTTP base URL (shard): {base_pick}", file=sys.stderr)
            url_cycle = None
        else:
            base_pick = None
            url_cycle = itertools.cycle(tei_all)
            tqdm.write(
                f"TEI HTTP: round-robin across {len(tei_all)} base URL(s)",
                file=sys.stderr,
            )

        headers = _tei_http_headers()
        with httpx.Client(timeout=timeout, headers=headers) as client:
            last_pmid: int | None = None
            with tqdm(total=pending_count, desc=pbar_desc, unit="pmid") as pbar:
                while not STOP_EVENT.is_set():
                    batch = _fetch_pending_article_batch(
                        articles_conn,
                        limit=args.limit,
                        shard_id=shard_id,
                        num_shards=num_shards,
                        last_pmid=last_pmid,
                        batch_size=embed_batch_size,
                        include_canonical_state=include_canonical_state,
                    )
                    if not batch:
                        break
                    last_pmid = batch[-1][0]
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

                    tei_base = base_pick if base_pick is not None else next(url_cycle)
                    arr = _post_tei_embed(client, tei_base, inputs)
                    dim = int(arr.shape[1])

                    if index is None:
                        index = _new_index(dim)
                    elif _index_dim(index) != dim:
                        raise RuntimeError(
                            f"Embedding dim mismatch: index has {_index_dim(index)}, TEI returned {dim}"
                        )

                    ids_np = np.asarray(pmids, dtype=np.int64)
                    index.add_with_ids(arr, ids_np)
                    since_ck.extend(pmids)
                    processed_this_run += len(pmids)

                    pbar.update(len(batch))

                    if index is not None and len(since_ck) >= checkpoint_every:
                        checkpoint(index)

    if source == "ollama":
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            run_ollama_loop(client)
            if index is not None and since_ck:
                checkpoint(index)
    elif source == "local":
        run_local_loop()
        if index is not None and since_ck:
            checkpoint(index)
    else:
        assert source == "tei-http"
        run_tei_http_loop()
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
