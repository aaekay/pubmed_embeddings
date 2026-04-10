- [x] Create uv-based Python project scaffolding.
- [x] Implement PubMed downloader CLI for baseline and update files.
- [x] Add idempotent download logic (skip unchanged files).
- [x] Document usage and examples.
- [x] Run basic verification and capture results.

## Review

- Verified with `uv sync && uv run pubmed-download --limit 1`.
- Baseline test file downloaded: `pubmed26n0001.xml.gz`.
- Update test file downloaded: `pubmed26n1335.xml.gz`.
- Added `tqdm` dependency and switched to streamed concurrent downloads.
- Verified with `uv run pubmed-download --limit 2 --workers 8` (progress bars + successful saves).
- Verified idempotency by rerunning same command (all files skipped as unchanged).
- No linter errors reported for `src/pubmed_embeddings/downloader.py`, `README.md`, and `pyproject.toml`.
