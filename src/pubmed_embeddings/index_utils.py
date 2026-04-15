from __future__ import annotations

import pathlib
import sqlite3
from collections.abc import Mapping
from typing import Iterator

import faiss
import numpy as np


FLAT_INDEX_FILENAME = "vectors.faiss"
HNSW_INDEX_FILENAME = "vectors.hnsw.faiss"

DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 128


def load_index(index_path: pathlib.Path, *, mmap: bool = False) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. "
            "Build embeddings first with `pubmed-embed` and use the merged canonical index."
        )
    if index_path.stat().st_size == 0:
        raise RuntimeError(f"FAISS index is empty: {index_path}")
    if mmap:
        return faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
    return faiss.read_index(str(index_path))


def read_state_meta(state_path: pathlib.Path) -> dict[str, str]:
    if not state_path.exists():
        return {}
    conn = sqlite3.connect(state_path)
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    finally:
        conn.close()
    return {
        str(row[0]): str(row[1])
        for row in rows
        if row[0] is not None and row[1] is not None
    }


def upsert_state_meta(conn: sqlite3.Connection, items: Mapping[str, object]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [(str(key), str(value)) for key, value in items.items()],
    )


def _downcast_id_map(index: faiss.Index) -> faiss.IndexIDMap | faiss.IndexIDMap2:
    idx = faiss.downcast_index(index)
    if not isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        raise RuntimeError("Expected FAISS IndexIDMap / IndexIDMap2")
    return idx


def extract_flat_ids_and_vectors(
    index: faiss.Index,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _downcast_id_map(index)
    sub = faiss.downcast_index(idx.index)
    if not isinstance(sub, faiss.IndexFlatIP):
        raise RuntimeError("Expected canonical flat index backed by IndexFlatIP")
    ids = faiss.vector_to_array(idx.id_map).astype(np.int64, copy=False)
    if idx.ntotal == 0:
        return ids, np.empty((0, int(sub.d)), dtype=np.float32)
    vectors = np.asarray(sub.reconstruct_n(0, idx.ntotal), dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return ids, vectors


def extract_flat_ids(index: faiss.Index) -> np.ndarray:
    idx = _downcast_id_map(index)
    sub = faiss.downcast_index(idx.index)
    if not isinstance(sub, faiss.IndexFlatIP):
        raise RuntimeError("Expected canonical flat index backed by IndexFlatIP")
    return faiss.vector_to_array(idx.id_map).astype(np.int64, copy=False)


def flat_index_dim(index: faiss.Index) -> int:
    idx = _downcast_id_map(index)
    sub = faiss.downcast_index(idx.index)
    if not isinstance(sub, faiss.IndexFlatIP):
        raise RuntimeError("Expected canonical flat index backed by IndexFlatIP")
    return int(sub.d)


def iter_flat_vector_batches(
    index: faiss.Index,
    *,
    batch_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    idx = _downcast_id_map(index)
    sub = faiss.downcast_index(idx.index)
    if not isinstance(sub, faiss.IndexFlatIP):
        raise RuntimeError("Expected canonical flat index backed by IndexFlatIP")
    total = int(idx.ntotal)
    if total == 0:
        return
    chunk = max(1, int(batch_size))
    for start in range(0, total, chunk):
        count = min(chunk, total - start)
        batch = np.asarray(sub.reconstruct_n(start, count), dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        yield start, batch


def build_hnsw_index(
    ids: np.ndarray,
    vectors: np.ndarray,
    *,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> faiss.IndexIDMap2:
    if vectors.ndim != 2:
        raise RuntimeError(f"Expected 2D vectors array, got shape {vectors.shape}")
    dim = int(vectors.shape[1])
    base = faiss.IndexHNSWFlat(dim, int(m), faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = int(ef_construction)
    base.hnsw.efSearch = int(ef_search)
    index = faiss.IndexIDMap2(base)
    if len(ids) != 0:
        index.add_with_ids(
            np.asarray(vectors, dtype=np.float32),
            np.asarray(ids, dtype=np.int64),
        )
    return index


def set_hnsw_ef_search(index: faiss.Index, ef_search: int) -> None:
    idx = faiss.downcast_index(index)
    if isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        idx = faiss.downcast_index(idx.index)
    if not isinstance(idx, faiss.IndexHNSWFlat):
        raise RuntimeError("Expected FAISS IndexHNSWFlat")
    idx.hnsw.efSearch = int(ef_search)
