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
- Added `pubmed-extract` CLI: streams PubMed XML(.gz), writes PMID/title/abstract/year/journal to SQLite with batched commits; handles `DeleteCitation` in update files.
- Smoke test: `uv run pubmed-extract` against local `data/*.xml.gz` produced expected row counts and sample queries.
- Extract: default DB is `<data-dir>/pubmed.sqlite`; WAL + cache PRAGMAs; batched deletes; resume via `ingested_files` (skip completed files); `--no-resume` and `--fast` documented.
- Extract: one transaction per file (rollback on failure); corrupt gzip/XML triggers FTP re-download by basename; default deletes local `*.xml.gz` after success (`--keep-xml` to retain).
- Embeddings: `pubmed-embed` reads eligible `articles` rows, calls Ollama `/api/embeddings`, L2-normalizes vectors, stores per-model `IndexIDMap2`+`IndexFlatIP` in `data/embeddings/<slug>/vectors.faiss` with `state.sqlite` for resume; atomic checkpoint; SIGINT/SIGTERM with final checkpoint; `.env.example` documents Ollama settings.
- Stats: `pubmed-stats` writes `data/stats_report/` (or `--out-dir`) with `summary.txt`, `articles_per_year.png`, `word_count_histogram.png`; matplotlib Agg backend for headless runs.
