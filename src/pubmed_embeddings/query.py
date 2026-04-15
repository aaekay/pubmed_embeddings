from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import sys
from dataclasses import asdict, dataclass
from typing import Iterable

import faiss
import httpx
import numpy as np
from dotenv import load_dotenv

from pubmed_embeddings.embeddings import (
    _announce_local_embed_device,
    _clip_embedding_text,
    _fetch_ollama_embedding,
    _index_dim,
    _parse_embedding_source,
    _parse_tei_http_base_urls,
    _post_tei_embed,
    _resolve_local_sentence_transformer_model,
    _slugify_model,
    _tei_http_headers,
)
from pubmed_embeddings.index_utils import (
    DEFAULT_HNSW_EF_SEARCH,
    FLAT_INDEX_FILENAME,
    HNSW_INDEX_FILENAME,
    load_index,
    read_state_meta,
    set_hnsw_ef_search,
)


@dataclass(slots=True)
class SearchHit:
    rank: int
    pmid: int
    score: float
    title: str | None
    abstract_preview: str | None
    year: int | None
    journal: str | None


def _truncate_text(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars < 1:
        return None
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    if max_chars <= 3:
        return compact[:max_chars]
    clipped = compact[: max_chars - 3].rstrip()
    if " " in clipped:
        head, tail = clipped.rsplit(" ", 1)
        if head and len(tail) < max(8, max_chars // 4):
            clipped = head
    return clipped + "..."


def _fetch_articles_by_pmid(
    conn: sqlite3.Connection,
    pmids: list[int],
) -> dict[int, tuple[str | None, str | None, int | None, str | None]]:
    if not pmids:
        return {}
    placeholders = ",".join("?" for _ in pmids)
    rows = conn.execute(
        f"""
        SELECT pmid, title, abstract, year, journal
        FROM articles
        WHERE pmid IN ({placeholders})
        """,
        pmids,
    ).fetchall()
    return {
        int(row[0]): (
            str(row[1]) if row[1] is not None else None,
            str(row[2]) if row[2] is not None else None,
            int(row[3]) if row[3] is not None else None,
            str(row[4]) if row[4] is not None else None,
        )
        for row in rows
    }


def _search_hits(
    index: faiss.Index,
    query_vec: np.ndarray,
    articles_conn: sqlite3.Connection,
    *,
    top_k: int,
    abstract_chars: int,
) -> list[SearchHit]:
    k = max(1, int(top_k))
    scores, labels = index.search(query_vec, k)
    pmids = [int(pid) for pid in labels[0].tolist() if int(pid) >= 0]
    articles = _fetch_articles_by_pmid(articles_conn, pmids)

    hits: list[SearchHit] = []
    for rank, (pmid_raw, score_raw) in enumerate(zip(labels[0].tolist(), scores[0].tolist()), start=1):
        pmid = int(pmid_raw)
        if pmid < 0:
            continue
        title, abstract, year, journal = articles.get(pmid, (None, None, None, None))
        hits.append(
            SearchHit(
                rank=rank,
                pmid=pmid,
                score=float(score_raw),
                title=title,
                abstract_preview=_truncate_text(abstract, abstract_chars),
                year=year,
                journal=journal,
            )
        )
    return hits


def _parse_int_meta(meta: dict[str, str], key: str) -> int | None:
    raw = (meta.get(key) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _select_query_index(
    out_dir: pathlib.Path,
    state_meta: dict[str, str],
    *,
    flat_only: bool,
    hnsw_ef_search: int | None,
) -> tuple[faiss.Index, str, str | None]:
    flat_path = out_dir / FLAT_INDEX_FILENAME
    hnsw_path = out_dir / HNSW_INDEX_FILENAME

    if flat_only:
        return load_index(flat_path), "flat", None

    current_dim = _parse_int_meta(state_meta, "dim")
    current_ntotal = _parse_int_meta(state_meta, "ntotal")
    built_dim = _parse_int_meta(state_meta, "hnsw_built_from_dim")
    built_ntotal = _parse_int_meta(state_meta, "hnsw_built_from_ntotal")
    meta_pref = (state_meta.get("query_index_type") or "").strip().lower()

    if hnsw_path.exists():
        hnsw_reason: str | None = None
        if built_dim is None or built_ntotal is None:
            hnsw_reason = "HNSW sidecar metadata is missing"
        elif current_dim is not None and built_dim != current_dim:
            hnsw_reason = (
                f"HNSW sidecar dim {built_dim} != canonical dim {current_dim}"
            )
        elif current_ntotal is not None and built_ntotal != current_ntotal:
            hnsw_reason = (
                f"HNSW sidecar ntotal {built_ntotal} != canonical ntotal {current_ntotal}"
            )
        elif meta_pref and meta_pref != "hnsw":
            hnsw_reason = (
                f"state metadata prefers {meta_pref!r}, indicating the canonical flat index changed after HNSW build"
            )

        if hnsw_reason is None:
            try:
                hnsw_index = load_index(hnsw_path)
                ef_search = (
                    hnsw_ef_search
                    if hnsw_ef_search is not None
                    else _parse_int_meta(state_meta, "hnsw_ef_search")
                    or DEFAULT_HNSW_EF_SEARCH
                )
                set_hnsw_ef_search(hnsw_index, ef_search)
                return hnsw_index, "hnsw", None
            except Exception as exc:
                hnsw_reason = f"failed to load HNSW sidecar ({exc})"

        if flat_path.exists():
            return (
                load_index(flat_path),
                "flat",
                f"HNSW sidecar present but not usable: {hnsw_reason}. Falling back to flat index.",
            )
        raise RuntimeError(f"HNSW sidecar present but not usable: {hnsw_reason}.")

    return load_index(flat_path), "flat", None


def _embed_query_ollama(
    *,
    model: str,
    query_text: str,
    base_url: str,
) -> np.ndarray:
    timeout = httpx.Timeout(600.0)
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        return _fetch_ollama_embedding(client, model, query_text)


def _embed_query_local(
    *,
    model: str,
    query_text: str,
    local_device: str | None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = _announce_local_embed_device(local_device)
    hf_model_id = _resolve_local_sentence_transformer_model(model)
    st_model = SentenceTransformer(hf_model_id, device=device)
    emb = st_model.encode(
        [query_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr


def _embed_query_tei_http(query_text: str) -> np.ndarray:
    urls = _parse_tei_http_base_urls()
    if not urls:
        raise RuntimeError("Set TEI_BASE_URL or TEI_BASE_URLS for EMBEDDING_SOURCE=tei-http.")
    timeout = httpx.Timeout(600.0)
    headers = _tei_http_headers()
    with httpx.Client(timeout=timeout, headers=headers) as client:
        return _post_tei_embed(client, urls[0], [query_text])


def _embed_query(
    *,
    source: str,
    model: str,
    query_text: str,
    ollama_base_url: str,
    local_device: str | None,
) -> np.ndarray:
    cleaned = _clip_embedding_text(query_text)
    if not cleaned:
        raise RuntimeError("Empty query after sanitization.")
    if source == "ollama":
        return _embed_query_ollama(
            model=model,
            query_text=cleaned,
            base_url=ollama_base_url,
        )
    if source == "local":
        return _embed_query_local(
            model=model,
            query_text=cleaned,
            local_device=local_device,
        )
    if source == "tei-http":
        return _embed_query_tei_http(cleaned)
    raise ValueError(f"Unsupported embedding source: {source!r}")


def _render_text_results(hits: list[SearchHit], *, index_type: str) -> str:
    header = f"Index: {index_type}"
    if not hits:
        return f"{header}\nNo results."

    lines: list[str] = [header, ""]
    for hit in hits:
        lines.append(f"{hit.rank}. PMID {hit.pmid}  score={hit.score:.4f}")
        meta_parts: list[str] = []
        if hit.year is not None:
            meta_parts.append(str(hit.year))
        if hit.journal:
            meta_parts.append(hit.journal)
        if meta_parts:
            lines.append("   " + " | ".join(meta_parts))
        if hit.title:
            lines.append(f"   Title: {hit.title}")
        if hit.abstract_preview:
            lines.append(f"   Abstract: {hit.abstract_preview}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embed a free-text query with the configured backend and search the merged canonical "
            "PubMed FAISS index."
        )
    )
    p.add_argument(
        "query",
        nargs="+",
        help="Free-text query to embed and search for.",
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
        help="Embedding output directory (default: <data-dir>/embeddings/<model-slug>)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name / id used to build the index; overrides EMBEDDING_MODEL",
    )
    p.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="Ollama base URL (overrides OLLAMA_BASE_URL)",
    )
    p.add_argument(
        "--local-device",
        type=str,
        default=None,
        help="Torch device for local query embeddings: auto, cuda, cpu, mps",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of hits to return (default: 10)",
    )
    p.add_argument(
        "--abstract-chars",
        type=int,
        default=280,
        help="Characters of abstract preview to print per hit (default: 280; use 0 to omit).",
    )
    p.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=None,
        help="Override runtime HNSW efSearch when the HNSW sidecar is used.",
    )
    p.add_argument(
        "--flat-only",
        action="store_true",
        help="Force exact search against the canonical flat index even if an HNSW sidecar exists.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    data_dir = args.data_dir.resolve()
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

    slug = _slugify_model(model)
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else data_dir / "embeddings" / slug
    ).resolve()
    state_path = out_dir / "state.sqlite"

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    query_text = " ".join(args.query).strip()
    if not query_text:
        print("Query text is empty.", file=sys.stderr)
        return 2

    try:
        state_meta = read_state_meta(state_path)
        stored_source = state_meta.get("embedding_source")
        if stored_source and stored_source != source:
            print(
                f"Embedding source mismatch: index metadata says {stored_source!r}, "
                f"but query is configured for {source!r}.",
                file=sys.stderr,
            )
            return 1
        index, index_type, index_warning = _select_query_index(
            out_dir,
            state_meta,
            flat_only=args.flat_only,
            hnsw_ef_search=args.hnsw_ef_search,
        )
        if index_warning:
            print(index_warning, file=sys.stderr)
        query_vec = _embed_query(
            source=source,
            model=model,
            query_text=query_text,
            ollama_base_url=args.ollama_base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434",
            local_device=args.local_device,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dim = _index_dim(index)
    if query_vec.shape[1] != dim:
        print(
            f"Embedding dim mismatch: index has {dim}, query encoder returned {query_vec.shape[1]}",
            file=sys.stderr,
        )
        return 1

    articles_conn = sqlite3.connect(db_path)
    try:
        hits = _search_hits(
            index,
            query_vec,
            articles_conn,
            top_k=max(1, args.top_k),
            abstract_chars=max(0, args.abstract_chars),
        )
    finally:
        articles_conn.close()

    if args.json:
        print(
            json.dumps(
                {
                    "index_type": index_type,
                    "hits": [asdict(hit) for hit in hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_render_text_results(hits, index_type=index_type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
