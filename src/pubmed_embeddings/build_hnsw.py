from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable

from dotenv import load_dotenv

from pubmed_embeddings.embeddings import _atomic_write_index, _connect_state, _slugify_model
from pubmed_embeddings.index_utils import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    FLAT_INDEX_FILENAME,
    HNSW_INDEX_FILENAME,
    build_hnsw_index,
    extract_flat_ids_and_vectors,
    load_index,
    upsert_state_meta,
)


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
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

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

    data_dir = args.data_dir.resolve()
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else data_dir / "embeddings" / _slugify_model(model)
    ).resolve()
    flat_path = out_dir / FLAT_INDEX_FILENAME
    hnsw_path = out_dir / HNSW_INDEX_FILENAME
    state_path = out_dir / "state.sqlite"

    if hnsw_path.exists() and not args.force:
        print(
            f"HNSW sidecar already exists: {hnsw_path}. Use --force to overwrite it.",
            file=sys.stderr,
        )
        return 1

    try:
        flat_index = load_index(flat_path)
        ids, vectors = extract_flat_ids_and_vectors(flat_index)
        hnsw_index = build_hnsw_index(
            ids,
            vectors,
            m=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _atomic_write_index(hnsw_index, hnsw_path)
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
                "hnsw_built_from_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            },
        )
        state_conn.commit()
    finally:
        state_conn.close()

    print(
        f"Built HNSW sidecar: {hnsw_path} "
        f"({hnsw_index.ntotal} vectors, M={args.m}, efConstruction={args.ef_construction}, efSearch={args.ef_search})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
