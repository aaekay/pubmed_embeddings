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

**Only the SQLite database is kept by default:** after a file is successfully ingested, the local `*.xml.gz` is deleted so you are not storing the full PubMed dumps on disk. Use `--keep-xml` to retain the archives (for example if you want to re-parse without re-downloading).

**Corrupt or unreadable gzip/XML:** If parsing fails in a way that usually indicates a bad archive (bad gzip, zlib error, XML parse error), the extractor re-downloads that file from NCBI FTP (`baseline` first, then `updatefiles`) using the same basename (e.g. `pubmed26n0001.xml.gz`) and retries up to `5` times. Filenames must match a real file on the FTP server (otherwise redownload fails with 404).

```bash
uv run pubmed-extract --data-dir data
```

Keep the downloaded `.xml.gz` files after ingest:

```bash
uv run pubmed-extract --data-dir data --keep-xml
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

## Embeddings (Ollama or local Sentence-Transformers + FAISS)

Build **one FAISS index per embedding model** (different models have different vector dimensions, so indexes are not mixed). Each run uses rows from `articles` where **both** `title` and `abstract` are present; the embedded text is `title` + space + `abstract`.

Set **`EMBEDDING_SOURCE`** to **`ollama`** (remote HTTP) or **`local`** (in-process [Sentence-Transformers](https://www.sbert.net/), no separate server). Legacy values **`tei`** / **`tie`** are accepted and mapped to **`local`**.

| Variable | When | Purpose |
|----------|------|---------|
| `EMBEDDING_SOURCE` | Always | `ollama` (default) or `local` |
| `OLLAMA_BASE_URL` | Ollama | e.g. `http://127.0.0.1:11434` |
| `LOCAL_EMBED_BATCH_SIZE` | Local | Articles per encode batch (default `32`; override with `--local-batch-size`) |
| `EMBEDDING_MODEL` | Both | Ollama model name (`ollama pull …`), or Hugging Face model id for local |

### Ollama

1. Copy [`.env.example`](.env.example) to `.env` and set `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_SOURCE=ollama`.
2. Pull the model in Ollama, e.g. `ollama pull bge-m3` or `ollama pull bge-large-en-v1.5`.
3. Run:

```bash
uv run pubmed-embed --data-dir data
```

`pubmed-embed` calls Ollama’s [`POST /api/embed`](https://docs.ollama.com/api/embed) with `input` and `truncate` (not the deprecated `/api/embeddings` + `prompt` path). Older Ollama builds that only expose `/api/embeddings` are tried if `/api/embed` returns 404.

### Local (Sentence-Transformers)

1. Ensure dependencies are installed (`uv sync` includes `sentence-transformers` and PyTorch).
2. Set `EMBEDDING_SOURCE=local` and `EMBEDDING_MODEL` to a Hugging Face sentence-embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`). The first run may download weights into the Hugging Face cache.
3. Optionally tune `LOCAL_EMBED_BATCH_SIZE` or `--local-batch-size` for throughput vs memory.

```bash
uv run pubmed-embed --data-dir data
```

Embeddings are computed in the same Python process; vectors are L2-normalized before indexing (same as the Ollama path).

Output layout (example for `bge-m3`):

- `data/embeddings/bge-m3/vectors.faiss` — L2-normalized vectors in `IndexIDMap2` over `IndexFlatIP` (cosine via inner product)
- `data/embeddings/bge-m3/state.sqlite` — `embedded_pmids` for resume; PMIDs are also reconciled from the FAISS id map on startup

**Second model** (e.g. `bge-large-en-v1.5`): change `EMBEDDING_MODEL` or pass `--model bge-large-en-v1.5` so output goes under `data/embeddings/bge-large-en-v1.5/` (separate index and state).

**Checkpointing:** every `--checkpoint-every` PMIDs (default `50`, or `EMBEDDING_CHECKPOINT_EVERY`), the index is written atomically (`*.tmp` then replace) and PMIDs are recorded in `state.sqlite`.

**Stop and resume:** Ctrl+C sets a stop flag; after the current request, a final checkpoint runs. Re-run the same command to continue; already embedded PMIDs are skipped.

**Testing:** `uv run pubmed-embed --limit 10` processes only the first 10 eligible rows.

## Statistics report (charts + summary)

Summarize the `articles` table: counts by year, missing title/abstract, both missing / both present, and a **histogram** of combined word counts (title + abstract, split on whitespace). Outputs go to a folder (default `data/stats_report/`): `summary.txt`, `articles_per_year.png`, `word_count_histogram.png`.

```bash
uv run pubmed-stats --data-dir data
```

- **Missing** means `NULL` or empty/whitespace-only text.
- Word histogram uses a capped x-axis by default (99.5th percentile, at least 500 words) so long tails do not squash the chart; override with `--word-max`.

```bash
uv run pubmed-stats --word-bins 40 --word-max 2000
```
