from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys
import time
from typing import Iterable

import faiss
from dotenv import load_dotenv
from tqdm import tqdm

from pubmed_embeddings.embeddings import _atomic_write_index, _connect_state, _slugify_model
from pubmed_embeddings.index_utils import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    FLAT_INDEX_FILENAME,
    HNSW_INDEX_FILENAME,
    extract_flat_ids,
    flat_index_dim,
    iter_flat_vector_batches,
    load_index,
    upsert_state_meta,
)

DEFAULT_ADD_BATCH_SIZE = 10_000


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a query-time HNSW sidecar from the canonical flat PubMed FAISS index."
        )
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
        help="Model name / id used to build the canonical flat index; overrides EMBEDDING_MODEL",
    )
    p.add_argument(
        "--m",
        type=int,
        default=DEFAULT_HNSW_M,
        help=f"HNSW M parameter (default: {DEFAULT_HNSW_M})",
    )
    p.add_argument(
        "--ef-construction",
        type=int,
        default=DEFAULT_HNSW_EF_CONSTRUCTION,
        help=f"HNSW efConstruction parameter (default: {DEFAULT_HNSW_EF_CONSTRUCTION})",
    )
    p.add_argument(
        "--ef-search",
        type=int,
        default=DEFAULT_HNSW_EF_SEARCH,
        help=f"Default HNSW efSearch parameter to store and use at runtime (default: {DEFAULT_HNSW_EF_SEARCH})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing HNSW sidecar if present.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ADD_BATCH_SIZE,
        help=f"Vectors to reconstruct/add per batch while building HNSW (default: {DEFAULT_ADD_BATCH_SIZE})",
    )
    return p.parse_args(argv)


def _log(message: str) -> None:
    tqdm.write(message, file=sys.stderr)


def _build_hnsw_sidecar_with_progress(
    ids,
    flat_index,
    *,
    dim: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    model_slug: str,
    batch_size: int,
) -> faiss.IndexIDMap2:
    base = faiss.IndexHNSWFlat(dim, int(m), faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = int(ef_construction)
    base.hnsw.efSearch = int(ef_search)
    index = faiss.IndexIDMap2(base)
    total = int(len(ids))
    if total == 0:
        return index

    desc = f"hnsw [{model_slug}]"
    with tqdm(total=total, desc=desc, unit="vec", file=sys.stderr) as pbar:
        for start, batch in iter_flat_vector_batches(flat_index, batch_size=batch_size):
            end = start + len(batch)
            index.add_with_ids(batch, ids[start:end])
            pbar.update(len(batch))
    return index


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    total_t0 = time.monotonic()

    model = (args.model or os.environ.get("EMBEDDING_MODEL") or "").strip()
    if not model:
        print("Set EMBEDDING_MODEL in .env or pass --model", file=sys.stderr)
        return 2
    if args.m < 2:
        print("--m must be >= 2", file=sys.stderr)
        return 2
    if args.ef_construction < 2:
        print("--ef-construction must be >= 2", file=sys.stderr)
        return 2
    if args.ef_search < 1:
        print("--ef-search must be >= 1", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2

    data_dir = args.data_dir.resolve()
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else data_dir / "embeddings" / _slugify_model(model)
    ).resolve()
    model_slug = out_dir.name
    flat_path = out_dir / FLAT_INDEX_FILENAME
    hnsw_path = out_dir / HNSW_INDEX_FILENAME
    state_path = out_dir / "state.sqlite"

    _log(f"Building HNSW sidecar for model={model!r}")
    _log(f"Canonical flat index: {flat_path}")
    _log(f"HNSW sidecar output: {hnsw_path}")
    _log(
        f"HNSW parameters: M={args.m}, efConstruction={args.ef_construction}, efSearch={args.ef_search}"
    )
    _log(f"Build batch size: {args.batch_size} vectors")

    if hnsw_path.exists() and not args.force:
        print(
            f"HNSW sidecar already exists: {hnsw_path}. Use --force to overwrite it.",
            file=sys.stderr,
        )
        return 1
    if hnsw_path.exists() and args.force:
        _log(f"Overwriting existing HNSW sidecar: {hnsw_path}")

    try:
        load_t0 = time.monotonic()
        _log("Loading canonical flat index with mmap ...")
        flat_index = load_index(flat_path, mmap=True)
        ids = extract_flat_ids(flat_index)
        dim = flat_index_dim(flat_index)
        load_dt = time.monotonic() - load_t0
        _log(
            f"Loaded {len(ids)} vector ids (dim={dim}) from canonical flat index in {load_dt:.2f}s"
        )
        _log(
            "Streaming vector reconstruction from the canonical flat index to reduce peak RAM."
        )

        build_t0 = time.monotonic()
        _log("Building HNSW graph ...")
        hnsw_index = _build_hnsw_sidecar_with_progress(
            ids,
            flat_index,
            dim=dim,
            m=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            model_slug=model_slug,
            batch_size=args.batch_size,
        )
        build_dt = time.monotonic() - build_t0
        _log(f"Built HNSW graph in {build_dt:.2f}s")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    del flat_index
    gc.collect()

    write_t0 = time.monotonic()
    _log(f"Writing HNSW sidecar to {hnsw_path} ...")
    _atomic_write_index(hnsw_index, hnsw_path)
    write_dt = time.monotonic() - write_t0
    _log(f"Wrote HNSW sidecar in {write_dt:.2f}s")

    meta_t0 = time.monotonic()
    _log(f"Updating state metadata in {state_path} ...")
    state_conn = _connect_state(state_path)
    try:
        upsert_state_meta(
            state_conn,
            {
                "query_index_type": "hnsw",
                "hnsw_m": args.m,
                "hnsw_ef_construction": args.ef_construction,
                "hnsw_ef_search": args.ef_search,
                "hnsw_built_from_ntotal": int(hnsw_index.ntotal),
                "hnsw_built_from_dim": dim,
            },
        )
        state_conn.commit()
    finally:
        state_conn.close()
    meta_dt = time.monotonic() - meta_t0
    total_dt = time.monotonic() - total_t0
    _log(f"Updated state metadata in {meta_dt:.2f}s")

    print(
        f"Built HNSW sidecar: {hnsw_path} "
        f"({hnsw_index.ntotal} vectors, M={args.m}, efConstruction={args.ef_construction}, "
        f"efSearch={args.ef_search}, total={total_dt:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
