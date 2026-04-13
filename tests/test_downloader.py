from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

from pubmed_embeddings.downloader import RemoteFile, _build_jobs


class DownloaderJobTests(unittest.TestCase):
    def test_existing_file_is_redownloaded_when_remote_size_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = pathlib.Path(tmp)
            existing = target_dir / "pubmed.xml.gz"
            existing.write_bytes(b"existing")
            remote_files = [
                RemoteFile(name="pubmed.xml.gz", url="https://example.com/pubmed.xml.gz")
            ]

            with mock.patch(
                "pubmed_embeddings.downloader.remote_size_bytes",
                return_value=None,
            ):
                jobs, skipped = _build_jobs(remote_files, target_dir=target_dir, force=False)

        self.assertEqual(skipped, 0)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].target_path.name, "pubmed.xml.gz")
        self.assertIsNone(jobs[0].expected_size)

    def test_existing_file_is_skipped_when_remote_size_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = pathlib.Path(tmp)
            existing = target_dir / "pubmed.xml.gz"
            existing.write_bytes(b"existing")
            remote_files = [
                RemoteFile(name="pubmed.xml.gz", url="https://example.com/pubmed.xml.gz")
            ]

            with mock.patch(
                "pubmed_embeddings.downloader.remote_size_bytes",
                return_value=existing.stat().st_size,
            ):
                jobs, skipped = _build_jobs(remote_files, target_dir=target_dir, force=False)

        self.assertEqual(skipped, 1)
        self.assertEqual(jobs, [])


if __name__ == "__main__":
    unittest.main()
