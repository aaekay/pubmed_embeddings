from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import unittest

from pubmed_embeddings import stats_report


class StatsReportTests(unittest.TestCase):
    def test_stats_report_writes_expected_summary_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            db_path = root / "pubmed.sqlite"
            out_dir = root / "stats"

            conn = sqlite3.connect(db_path)
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
                    (1, "title one", "abstract one", 2020, "j"),
                    (2, "  ", "abstract two", 2021, "j"),
                    (3, None, None, None, "j"),
                ],
            )
            conn.commit()
            conn.close()

            rc = stats_report.main(
                ["--db", str(db_path), "--out-dir", str(out_dir), "--word-bins", "10"]
            )

            self.assertEqual(rc, 0)
            summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Total articles: 3", summary)
            self.assertIn("Title missing: 2", summary)
            self.assertIn("Abstract missing: 1", summary)
            self.assertIn("Both title and abstract missing: 1", summary)
            self.assertIn("Both title and abstract present: 1", summary)
            self.assertTrue((out_dir / "articles_per_year.png").exists())
            self.assertTrue((out_dir / "word_count_histogram.png").exists())


if __name__ == "__main__":
    unittest.main()
