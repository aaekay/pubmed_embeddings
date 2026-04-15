from __future__ import annotations

import io
import os
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import faiss
import numpy as np

from pubmed_embeddings.build_hnsw import main
from pubmed_embeddings.index_utils import (
    HNSW_INDEX_FILENAME,
    iter_flat_vector_batches,
    load_index as real_load_index,
    read_state_meta,
)


class BuildHnswTests(unittest.TestCase):
    def test_main_builds_sidecar_and_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            out_dir = root / "embeddings" / "test-model"
            out_dir.mkdir(parents=True, exist_ok=True)

            flat = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
            vecs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            faiss.normalize_L2(vecs)
            flat.add_with_ids(vecs, np.asarray([11, 22], dtype=np.int64))
            faiss.write_index(flat, str(out_dir / "vectors.faiss"))

            state_conn = sqlite3.connect(out_dir / "state.sqlite")
            state_conn.execute(
                """
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            state_conn.commit()
            state_conn.close()

            stdout = io.StringIO()
            stderr = io.StringIO()
            old_env = os.environ.copy()
            os.environ["EMBEDDING_MODEL"] = "test-model"
            try:
                with (
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                    mock.patch("pubmed_embeddings.build_hnsw.load_index", wraps=real_load_index) as load_index_mock,
                ):
                    rc = main(["--out-dir", str(out_dir), "--batch-size", "1"])
            finally:
                os.environ.clear()
                os.environ.update(old_env)

            self.assertEqual(rc, 0)
            load_index_mock.assert_called_once()
            call_args, call_kwargs = load_index_mock.call_args
            self.assertEqual(pathlib.Path(call_args[0]).name, "vectors.faiss")
            self.assertTrue(call_kwargs["mmap"])
            self.assertTrue((out_dir / HNSW_INDEX_FILENAME).exists())
            meta = read_state_meta(out_dir / "state.sqlite")
            self.assertEqual(meta["query_index_type"], "hnsw")
            self.assertEqual(meta["hnsw_built_from_ntotal"], "2")
            self.assertEqual(meta["hnsw_built_from_dim"], "2")
            self.assertIn("Built HNSW sidecar", stdout.getvalue())
            err = stderr.getvalue()
            self.assertIn("Loading canonical flat index", err)
            self.assertIn("Build batch size: 1 vectors", err)
            self.assertIn("Loaded 2 vector ids", err)
            self.assertIn("Streaming vector reconstruction", err)
            self.assertIn("Building HNSW graph", err)
            self.assertIn("hnsw [test-model]", err)
            self.assertIn("Writing HNSW sidecar", err)
            self.assertIn("Updating state metadata", err)

    def test_iter_flat_vector_batches_splits_reconstruction(self) -> None:
        flat = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
        vecs = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
        faiss.normalize_L2(vecs)
        flat.add_with_ids(vecs, np.asarray([11, 22, 33], dtype=np.int64))

        batches = list(iter_flat_vector_batches(flat, batch_size=2))
        self.assertEqual([start for start, _ in batches], [0, 2])
        self.assertEqual([batch.shape for _, batch in batches], [(2, 2), (1, 2)])
        np.testing.assert_allclose(batches[0][1][0], vecs[0], atol=1e-6)
        np.testing.assert_allclose(batches[1][1][0], vecs[2], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
