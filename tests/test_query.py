from __future__ import annotations

import io
import json
import os
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import faiss
import numpy as np

from pubmed_embeddings.index_utils import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    build_hnsw_index,
    extract_flat_ids_and_vectors,
)
from pubmed_embeddings.query import _render_text_results, _search_hits, main


class QueryRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self.db_path = root / "pubmed.sqlite"

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE articles (
                pmid INTEGER PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                year INTEGER,
                journal TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO articles (pmid, title, abstract, year, journal) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    101,
                    "Cancer immunotherapy biomarkers",
                    "Biomarker-guided immunotherapy improves response selection in cancer cohorts.",
                    2024,
                    "Journal A",
                ),
                (
                    202,
                    "Cardiology outcomes study",
                    "Cardiology endpoints and follow-up outcomes are summarized for a separate cohort.",
                    2023,
                    "Journal B",
                ),
            ],
        )
        conn.commit()
        conn.close()

        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
        vecs = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )
        faiss.normalize_L2(vecs)
        ids = np.asarray([101, 202, 303], dtype=np.int64)
        self.index.add_with_ids(vecs, ids)
        self.out_dir = root / "embeddings" / "test-model"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.out_dir / "vectors.faiss"))
        self.state_path = self.out_dir / "state.sqlite"
        self._write_meta(
            {
                "embedding_source": "local",
                "dim": 2,
                "ntotal": 3,
                "query_index_type": "flat",
            }
        )

    def _write_meta(self, items: dict[str, object]) -> None:
        state_conn = sqlite3.connect(self.state_path)
        state_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        state_conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [(str(key), str(value)) for key, value in items.items()],
        )
        state_conn.commit()
        state_conn.close()

    def _write_hnsw_sidecar(self, *, query_index_type: str = "hnsw", ntotal: int = 3) -> None:
        ids, vectors = extract_flat_ids_and_vectors(self.index)
        hnsw = build_hnsw_index(
            ids,
            vectors,
            m=DEFAULT_HNSW_M,
            ef_construction=DEFAULT_HNSW_EF_CONSTRUCTION,
            ef_search=DEFAULT_HNSW_EF_SEARCH,
        )
        faiss.write_index(hnsw, str(self.out_dir / "vectors.hnsw.faiss"))
        self._write_meta(
            {
                "query_index_type": query_index_type,
                "hnsw_m": DEFAULT_HNSW_M,
                "hnsw_ef_construction": DEFAULT_HNSW_EF_CONSTRUCTION,
                "hnsw_ef_search": DEFAULT_HNSW_EF_SEARCH,
                "hnsw_built_from_ntotal": ntotal,
                "hnsw_built_from_dim": 2,
            }
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_search_hits_rank_and_hydrate_metadata(self) -> None:
        query_vec = np.asarray([[1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        conn = sqlite3.connect(self.db_path)
        try:
            hits = _search_hits(
                self.index,
                query_vec,
                conn,
                top_k=3,
                abstract_chars=60,
            )
        finally:
            conn.close()

        self.assertEqual([hit.pmid for hit in hits], [101, 303, 202])
        self.assertEqual(hits[0].title, "Cancer immunotherapy biomarkers")
        self.assertEqual(hits[0].year, 2024)
        self.assertEqual(hits[1].title, None)
        self.assertEqual(hits[1].journal, None)
        self.assertTrue(hits[0].score > hits[1].score > hits[2].score)

    def test_render_text_results_formats_hits_and_truncates_abstract(self) -> None:
        query_vec = np.asarray([[1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        conn = sqlite3.connect(self.db_path)
        try:
            hits = _search_hits(
                self.index,
                query_vec,
                conn,
                top_k=1,
                abstract_chars=35,
            )
        finally:
            conn.close()

        rendered = _render_text_results(hits, index_type="flat")
        self.assertIn("Index: flat", rendered)
        self.assertIn("1. PMID 101", rendered)
        self.assertIn("2024 | Journal A", rendered)
        self.assertIn("Title: Cancer immunotherapy biomarkers", rendered)
        self.assertIn("Abstract: Biomarker-guided immunotherapy...", rendered)

    def test_main_outputs_json_hits(self) -> None:
        query_vec = np.asarray([[1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        stdout = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "EMBEDDING_MODEL": "test-model",
                    "EMBEDDING_SOURCE": "local",
                },
                clear=False,
            ),
            mock.patch("pubmed_embeddings.query._embed_query", return_value=query_vec),
            redirect_stdout(stdout),
        ):
            rc = main(
                [
                    "--db",
                    str(self.db_path),
                    "--out-dir",
                    str(self.out_dir),
                    "--top-k",
                    "2",
                    "--json",
                    "cancer biomarkers",
                ]
            )

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["index_type"], "flat")
        self.assertEqual([row["pmid"] for row in payload["hits"]], [101, 303])
        self.assertEqual(payload["hits"][0]["title"], "Cancer immunotherapy biomarkers")
        self.assertEqual(payload["hits"][1]["title"], None)

    def test_main_prefers_hnsw_sidecar(self) -> None:
        query_vec = np.asarray([[1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        self._write_hnsw_sidecar()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "EMBEDDING_MODEL": "test-model",
                    "EMBEDDING_SOURCE": "local",
                },
                clear=False,
            ),
            mock.patch("pubmed_embeddings.query._embed_query", return_value=query_vec),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            rc = main(
                [
                    "--db",
                    str(self.db_path),
                    "--out-dir",
                    str(self.out_dir),
                    "--json",
                    "cancer biomarkers",
                ]
            )

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["index_type"], "hnsw")
        self.assertEqual([row["pmid"] for row in payload["hits"][:2]], [101, 303])
        self.assertEqual(stderr.getvalue(), "")

    def test_main_falls_back_to_flat_when_hnsw_is_stale(self) -> None:
        query_vec = np.asarray([[1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        self._write_hnsw_sidecar(ntotal=2)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "EMBEDDING_MODEL": "test-model",
                    "EMBEDDING_SOURCE": "local",
                },
                clear=False,
            ),
            mock.patch("pubmed_embeddings.query._embed_query", return_value=query_vec),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            rc = main(
                [
                    "--db",
                    str(self.db_path),
                    "--out-dir",
                    str(self.out_dir),
                    "--json",
                    "cancer biomarkers",
                ]
            )

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["index_type"], "flat")
        self.assertIn("Falling back to flat index", stderr.getvalue())

    def test_main_flat_only_ignores_valid_hnsw(self) -> None:
        query_vec = np.asarray([[1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        self._write_hnsw_sidecar()

        stdout = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "EMBEDDING_MODEL": "test-model",
                    "EMBEDDING_SOURCE": "local",
                },
                clear=False,
            ),
            mock.patch("pubmed_embeddings.query._embed_query", return_value=query_vec),
            redirect_stdout(stdout),
        ):
            rc = main(
                [
                    "--db",
                    str(self.db_path),
                    "--out-dir",
                    str(self.out_dir),
                    "--flat-only",
                    "--json",
                    "cancer biomarkers",
                ]
            )

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["index_type"], "flat")

    def test_main_rejects_embedding_source_mismatch(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "EMBEDDING_MODEL": "test-model",
                    "EMBEDDING_SOURCE": "ollama",
                },
                clear=False,
            ),
            mock.patch("pubmed_embeddings.query._embed_query") as embed_query,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            rc = main(
                [
                    "--db",
                    str(self.db_path),
                    "--out-dir",
                    str(self.out_dir),
                    "cancer biomarkers",
                ]
            )

        self.assertEqual(rc, 1)
        embed_query.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Embedding source mismatch", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
