- [x] Create uv-based Python project scaffolding.
- [x] Implement PubMed downloader CLI for baseline and update files.
- [x] Add idempotent download logic (skip unchanged files).
- [x] Document usage and examples.
- [x] Run basic verification and capture results.
- [x] Add a TEI cluster supervisor CLI that can build or reuse TEI and launch one server per GPU on ports 11450-11452.
- [x] Wire the new CLI into project packaging and align the embedding client with explicit `tei-http` usage.
- [x] Update `.env.example` and `README.md` to document the TEI cluster workflow and separate it from local in-process embeddings.
- [x] Add automated tests for TEI supervisor logic and embedding source validation.
- [x] Run verification and capture results for the TEI workflow changes.
- [x] Harden TEI supervisor binary probing, install prerequisites, archive validation, and startup cleanup.
- [x] Refactor embedding selection/resume logic to avoid full in-memory materialization before sharding.
- [x] Make downloader re-download when remote size cannot be confirmed instead of silently trusting local files.
- [x] Stream stats aggregation instead of loading all article rows into memory.
- [x] Expand tests for TEI process logic, embedding query behavior, downloader job selection, and stats output.
- [x] Run verification and capture results for the hardening/scalability pass.

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
- Embeddings: `pubmed-embed` reads eligible `articles` rows, calls Ollama `POST /api/embed` (`input` + `truncate`; fallback to legacy `/api/embeddings`), L2-normalizes vectors, stores per-model `IndexIDMap2`+`IndexFlatIP` in `data/embeddings/<slug>/vectors.faiss` with `state.sqlite` for resume; atomic checkpoint; SIGINT/SIGTERM with final checkpoint; `.env.example` documents Ollama settings.
- Embeddings (local): `EMBEDDING_SOURCE=local` uses in-process Sentence-Transformers (`LOCAL_EMBED_BATCH_SIZE` / `--local-batch-size`); `embedding_source` is stored in `state.sqlite` meta; README + `.env.example` updated.
- Stats: `pubmed-stats` writes `data/stats_report/` (or `--out-dir`) with `summary.txt`, `articles_per_year.png`, `word_count_histogram.png`; matplotlib Agg backend for headless runs.
- Added `pubmed-tei-cluster`: a foreground TEI supervisor that validates Linux/NVIDIA prerequisites, probes any `text-embeddings-router` found on `PATH`, reuses it only when the version/CLI surface matches the requested TEI release, otherwise falls back to a managed source build, launches one worker per GPU, waits for health checks, and prints the matching `pubmed-embed` command for TEI HTTP on ports `11450,11451,11452`.
- Embedding source handling is now explicit: `EMBEDDING_SOURCE=tei` / `tie` is rejected with a migration error; use `local` for in-process Sentence-Transformers or `tei-http` for TEI servers.
- Updated `.env.example` and `README.md` to document the new TEI supervisor workflow, the `tei-http` client configuration, and the fixed default ports `11450-11452`.
- Verification: `uv run python -m py_compile src/pubmed_embeddings/*.py tests/*.py`, `.venv/bin/python -m unittest discover -s tests -q`, `uv run pubmed-tei-cluster --help`, and `uv run pubmed-embed --help`.
- Hardening/scalability pass: removed automatic remote rustup bootstrap; `pubmed-tei-cluster` now requires an existing Rust toolchain, validates downloaded TEI archives before extract/build, and terminates already-started workers immediately if a later worker fails readiness.
- Embeddings: pending article selection is now SQL-backed and batched, with state/canonical-state filtering done through attached SQLite databases instead of full Python lists/sets before sharding.
- Downloader: local files are re-downloaded when remote size cannot be confirmed instead of being silently skipped as unchanged.
- Stats: aggregation now streams rows and keeps only the numeric word-count series needed for histogram/percentile calculation.
- Verification: `uv run python -m py_compile src/pubmed_embeddings/*.py tests/*.py`, `.venv/bin/python -m unittest discover -s tests -q`, `uv run pubmed-tei-cluster --help`, `uv run pubmed-embed --help`, `uv run pubmed-download --help`, and `uv run pubmed-stats --help`.

## Current Review Task

- [x] Inspect the current query and retrieval implementation paths.
- [x] Verify query behavior with targeted tests or runnable commands.
- [x] Summarize current query-system quality, limitations, and recommended next steps.

### Query Review Notes

- `pubmed-embed` currently builds and resumes FAISS indices, but there is no separate query/search CLI or API in `pyproject.toml` scripts.
- `src/pubmed_embeddings/embeddings.py` defines index creation, checkpointing, shard merge, and pending-article SQL selection, but no retrieval call site (`index.search(...)`) or query embedding flow.
- README documents index output and says downstream search should use the merged canonical index, but does not document an implemented query command.
- Automated coverage for “query” is limited to the SQL that selects pending articles for embedding in `tests/test_embeddings.py`; there are no semantic retrieval tests.
- Verification attempt: `uv run pytest -q tests/test_embeddings.py` initially failed because `PYTHONPATH=src` was missing; rerun with `PYTHONPATH=src` then failed because `faiss` is not importable in the current environment even though `faiss-cpu` is declared in `pyproject.toml`.

## Query CLI Implementation

- [x] Add a dedicated `pubmed-query` CLI entry point and module.
- [x] Reuse the existing embedding backends (`ollama`, `local`, `tei-http`) to embed a user query with the same model selection rules.
- [x] Search the merged canonical FAISS index and hydrate PMIDs back to article metadata from SQLite.
- [x] Add targeted tests for query retrieval and output formatting.
- [x] Update README and review notes with usage and verification results.

### Query CLI Review

- Added `pubmed-query` as a package entry point and implemented `src/pubmed_embeddings/query.py`.
- Query execution now embeds free-text input with the configured backend, searches the merged canonical `vectors.faiss`, and joins returned PMIDs back to SQLite article metadata.
- Added JSON output plus terminal-friendly text output with title, year, journal, score, and abstract preview.
- Added `tests/test_query.py` covering FAISS ranking, metadata hydration, preview formatting, and a mocked CLI-level JSON flow through `main()`.
- Verification: `python -m py_compile src/pubmed_embeddings/query.py tests/test_query.py`, `PYTHONPATH=src .venv/bin/python -m unittest -q tests.test_query tests.test_embeddings`, `PYTHONPATH=src .venv/bin/python -m pubmed_embeddings.query --help`, and `uv run pubmed-query --help`.

## HNSW Query Acceleration

- [x] Add an HNSW sidecar build CLI that reads the canonical flat index and writes `vectors.hnsw.faiss`.
- [x] Keep the flat index as the canonical write/resume artifact while recording enough metadata to detect stale HNSW sidecars.
- [x] Update `pubmed-query` to prefer HNSW when valid, allow flat-only queries, and fall back cleanly to flat when the sidecar is missing or stale.
- [x] Add tests for HNSW build, query-side selection/fallback, and output metadata.
- [x] Update README and review notes with the new HNSW workflow and verification results.

### HNSW Review

- Added shared FAISS index helpers in `src/pubmed_embeddings/index_utils.py`.
- Added `pubmed-build-hnsw` in `src/pubmed_embeddings/build_hnsw.py` to build `vectors.hnsw.faiss` from the canonical flat `vectors.faiss` and record HNSW metadata in `state.sqlite`.
- Flat index checkpoints and merges now store `ntotal` and `query_index_type=flat` in `state.sqlite`, which lets queries detect when a previously built HNSW sidecar is stale after canonical flat index changes.
- `pubmed-query` now prefers HNSW when the sidecar exists and matches the canonical flat metadata, supports `--hnsw-ef-search` and `--flat-only`, falls back to flat with a warning when HNSW is stale or unusable, and reports the selected index type in text and JSON output.
- Added `tests/test_build_hnsw.py` plus expanded `tests/test_query.py` coverage for HNSW selection, stale-sidecar fallback, flat-only behavior, and builder metadata.
- Verification: `python -m py_compile src/pubmed_embeddings/index_utils.py src/pubmed_embeddings/build_hnsw.py src/pubmed_embeddings/query.py tests/test_build_hnsw.py tests/test_query.py`, `PYTHONPATH=src .venv/bin/python -m unittest -q tests.test_build_hnsw tests.test_query tests.test_embeddings`, `uv run pubmed-build-hnsw --help`, and `uv run pubmed-query --help`.

## HNSW Builder Verbosity

- [x] Make `pubmed-build-hnsw` print stage-level progress and timings in the terminal.
- [x] Add test coverage for the more verbose builder output.
- [x] Run targeted verification for the builder UX changes.

### HNSW Builder Verbosity Review

- `pubmed-build-hnsw` now prints stage-level terminal output for model/path selection, flat-index loading, vector counts/dim, HNSW graph build start/end, sidecar write start/end, metadata update start/end, and total runtime.
- HNSW insertion now uses a tqdm progress bar (`hnsw [<model-slug>]`) so large builds show ongoing progress instead of appearing stalled.
- The final success line still prints to stdout, but the operational progress and timings are emitted to stderr for terminal visibility.
- Expanded `tests/test_build_hnsw.py` to assert the verbose builder output.
- Verification: `python -m py_compile src/pubmed_embeddings/build_hnsw.py tests/test_build_hnsw.py`, `PYTHONPATH=src .venv/bin/python -m unittest -q tests.test_build_hnsw tests.test_query tests.test_embeddings`, and `uv run pubmed-build-hnsw --help`.

## HNSW Builder Memory

- [x] Reduce peak RAM for `pubmed-build-hnsw` so large sidecar builds do not crash the server.
- [x] Verify the lower-memory build path with targeted tests.

### HNSW Builder Memory Review

- `pubmed-build-hnsw` now memory-maps the canonical flat `vectors.faiss` and reconstructs vectors batch-by-batch instead of materializing the full vector matrix in Python memory before HNSW insertion.
- Added `--batch-size` (default `10000`) so tight-memory servers can trade speed for lower temporary RAM during HNSW build.
- Builder logs now explicitly state that it is streaming vector reconstruction from the canonical flat index to reduce peak RAM.
- Expanded `tests/test_build_hnsw.py` to assert mmap loading, batch-oriented builder output, and the reconstruction batch helper.
- Verification: `python -m py_compile src/pubmed_embeddings/index_utils.py src/pubmed_embeddings/build_hnsw.py tests/test_build_hnsw.py` and `PYTHONPATH=src .venv/bin/python -m unittest -q tests.test_build_hnsw tests.test_query tests.test_embeddings`.
