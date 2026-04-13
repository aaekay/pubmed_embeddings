from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import unittest

from pubmed_embeddings.embeddings import (
    _connect_state,
    _count_pending_articles,
    _fetch_pending_article_batch,
    _parse_embedding_source,
    _prepare_pending_articles_connection,
)


class EmbeddingSourceTests(unittest.TestCase):
    def test_allows_supported_sources(self) -> None:
        self.assertEqual(_parse_embedding_source("ollama"), "ollama")
        self.assertEqual(_parse_embedding_source("local"), "local")
        self.assertEqual(_parse_embedding_source("tei-http"), "tei-http")

    def test_rejects_legacy_tie_aliases(self) -> None:
        with self.assertRaises(ValueError):
            _parse_embedding_source("tei")
        with self.assertRaises(ValueError):
            _parse_embedding_source("tie")


class PendingArticleQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self.db_path = root / "pubmed.sqlite"
        self.state_path = root / "state.sqlite"
        self.canonical_state_path = root / "canonical.sqlite"

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
                (1, "t1", "a1", 2020, "j"),
                (2, "t2", "a2", 2020, "j"),
                (3, "t3", "a3", 2020, "j"),
                (4, "t4", "a4", 2020, "j"),
                (5, "t5", "   ", 2020, "j"),
                (6, "t6", "a6", 2020, "j"),
            ],
        )
        conn.commit()
        conn.close()

        state_conn = _connect_state(self.state_path)
        state_conn.execute("INSERT INTO embedded_pmids (pmid) VALUES (2)")
        state_conn.commit()
        state_conn.close()

        canonical_conn = _connect_state(self.canonical_state_path)
        canonical_conn.execute("INSERT INTO embedded_pmids (pmid) VALUES (4)")
        canonical_conn.commit()
        canonical_conn.close()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_count_and_fetch_pending_articles(self) -> None:
        conn, include_canonical = _prepare_pending_articles_connection(
            self.db_path,
            state_path=self.state_path,
            canonical_state_path=self.canonical_state_path,
        )
        try:
            count = _count_pending_articles(
                conn,
                limit=None,
                shard_id=None,
                num_shards=None,
                include_canonical_state=include_canonical,
            )
            self.assertEqual(count, 3)

            batch1 = _fetch_pending_article_batch(
                conn,
                limit=None,
                shard_id=None,
                num_shards=None,
                last_pmid=None,
                batch_size=2,
                include_canonical_state=include_canonical,
            )
            batch2 = _fetch_pending_article_batch(
                conn,
                limit=None,
                shard_id=None,
                num_shards=None,
                last_pmid=batch1[-1][0],
                batch_size=2,
                include_canonical_state=include_canonical,
            )
        finally:
            conn.close()

        self.assertEqual([row[0] for row in batch1], [1, 3])
        self.assertEqual([row[0] for row in batch2], [6])

    def test_limit_applies_before_shard_and_resume_filters(self) -> None:
        conn, include_canonical = _prepare_pending_articles_connection(
            self.db_path,
            state_path=self.state_path,
            canonical_state_path=self.canonical_state_path,
        )
        try:
            count = _count_pending_articles(
                conn,
                limit=4,
                shard_id=1,
                num_shards=2,
                include_canonical_state=include_canonical,
            )
            batch = _fetch_pending_article_batch(
                conn,
                limit=4,
                shard_id=1,
                num_shards=2,
                last_pmid=None,
                batch_size=10,
                include_canonical_state=include_canonical,
            )
        finally:
            conn.close()

        self.assertEqual(count, 2)
        self.assertEqual([row[0] for row in batch], [1, 3])


if __name__ == "__main__":
    unittest.main()
