from __future__ import annotations

import io
import os
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import faiss
import numpy as np

from pubmed_embeddings.build_hnsw import main
from pubmed_embeddings.index_utils import HNSW_INDEX_FILENAME, read_state_meta


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
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main(["--out-dir", str(out_dir)])
            finally:
                os.environ.clear()
                os.environ.update(old_env)

            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / HNSW_INDEX_FILENAME).exists())
            meta = read_state_meta(out_dir / "state.sqlite")
            self.assertEqual(meta["query_index_type"], "hnsw")
            self.assertEqual(meta["hnsw_built_from_ntotal"], "2")
            self.assertEqual(meta["hnsw_built_from_dim"], "2")
            self.assertIn("Built HNSW sidecar", stdout.getvalue())
            err = stderr.getvalue()
            self.assertIn("Loading canonical flat index", err)
            self.assertIn("Loaded 2 vectors", err)
            self.assertIn("Building HNSW graph", err)
            self.assertIn("hnsw [test-model]", err)
            self.assertIn("Writing HNSW sidecar", err)
            self.assertIn("Updating state metadata", err)


if __name__ == "__main__":
    unittest.main()
