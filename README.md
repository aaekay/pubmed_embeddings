# PubMed Downloader

Download PubMed annual baseline and daily update XML archives into a local data directory.

## Setup (uv + venv)

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Usage

Run both baseline and updates:

```bash
uv run pubmed-download
```

Only baseline:

```bash
uv run pubmed-download --baseline
```

Only updates:

```bash
uv run pubmed-download --updates
```

Quick test run (2 files each source):

```bash
uv run pubmed-download --limit 2
```

Tune parallelism (faster downloads on good connections):

```bash
uv run pubmed-download --workers 12
```

Force re-download:

```bash
uv run pubmed-download --force
```

## Progress output

- `tqdm` file progress bar shows how many files were processed.
- `tqdm` byte progress bar shows streaming download throughput.
- Final summary prints `checked`, `saved`, `skipped`, and `failed`.

## Extract to SQLite

After downloading `pubmed*.xml.gz` files into `data/`, parse them into a local SQLite database (PMID, title, abstract, publication year, journal). Rows are inserted in batches as each file is streamed; `INSERT ... ON CONFLICT` upserts by PMID. Update files that contain `DeleteCitation` remove matching PMIDs from the database.

**Database location:** By default the DB is created at `<data-dir>/pubmed.sqlite` (for example `data/pubmed.sqlite` when `--data-dir` is `data`). Override with `--db /path/to/file.sqlite`.

```bash
uv run pubmed-extract --data-dir data
```

Specific files only:

```bash
uv run pubmed-extract --data-dir data --files data/pubmed26n0001.xml.gz
```

Tune batch size (default 500 rows or deletes per commit):

```bash
uv run pubmed-extract --batch-size 2000
```

Faster bulk ingest (uses `PRAGMA synchronous=OFF`; slightly higher risk if the machine loses power mid-write):

```bash
uv run pubmed-extract --fast
```

### Resumable ingestion

Each fully processed input file is recorded in the `ingested_files` table (basename + file size + modification time). If you stop the script and run it again, **completed files are skipped**. A file that was interrupted is **not** marked complete, so the next run re-reads it; upserts make that safe.

- Re-process everything (ignore resume state): `--no-resume`

### Speed notes

The extractor sets WAL mode, a large page cache, memory temp store, and `synchronous=NORMAL` by default. Larger `--batch-size` reduces commit overhead. For maximum throughput on a dedicated machine, use `--fast`.
