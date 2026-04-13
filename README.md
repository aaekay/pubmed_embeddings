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

Set **`EMBEDDING_SOURCE`** to **`ollama`** (remote HTTP), **`local`** (in-process [Sentence-Transformers](https://www.sbert.net/), no separate server), or **`tei-http`** ([Hugging Face Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) over HTTP). Legacy values **`tei`** / **`tie`** are accepted and mapped to **`local`** (in-process ST, not HTTP).

| Variable | When | Purpose |
|----------|------|---------|
| `EMBEDDING_SOURCE` | Always | `ollama` (default), `local`, or `tei-http` |
| `OLLAMA_BASE_URL` | Ollama | e.g. `http://127.0.0.1:11434` |
| `LOCAL_EMBED_BATCH_SIZE` | Local / tei-http | Articles per encode batch (default `32`; override with `--local-batch-size`) |
| `TEI_BATCH_SIZE` | Local / tei-http | Used as batch size when `LOCAL_EMBED_BATCH_SIZE` is unset (same as legacy TEI naming in `.env`) |
| `TEI_BASE_URL` | tei-http | Single TEI server base URL (no trailing slash), e.g. `http://127.0.0.1:8080` |
| `TEI_BASE_URLS` | tei-http | Comma-separated list for **round-robin** (single process) or **one URL per shard** when using `--workers` (shard `k` uses `urls[k % len(urls)]`) |
| `TEI_EMBED_PATH` | tei-http | Embed path (default `/embed`) |
| `TEI_API_KEY` / `HF_TOKEN` | tei-http | Optional `Authorization: Bearer` for hosted TEI |
| `TEI_TRUNCATE` | tei-http | `true`/`false` — passed as JSON `truncate` to TEI (default true) |
| `LOCAL_EMBED_FP16` | Local | If `1`/`true`/`yes` and `LOCAL_EMBED_DEVICE=cuda`, load the ST model in **fp16** (outputs cast to float32 for FAISS; validate similarity if you need bit-identical vectors) |
| `EMBEDDING_CHECKPOINT_EVERY` | All except default nuance | PMIDs between full `vectors.faiss` + `state.sqlite` checkpoints (CLI `--checkpoint-every` wins; default **5000** for local and tei-http, **50** for Ollama if unset) |
| `EMBED_NUM_WORKERS` | Local / tei-http | Default for `--workers` when not passed on the CLI |
| `EMBEDDING_MODEL` | All | Ollama model name (`ollama pull …`), Hugging Face id for local; for **tei-http** this names the output directory and metadata (the TEI server loads its own model) |

### Ollama

1. Copy [`.env.example`](.env.example) to `.env` and set `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_SOURCE=ollama`.
2. Pull the model in Ollama, e.g. `ollama pull bge-m3` or `ollama pull bge-large-en-v1.5`.
3. Run:

```bash
uv run pubmed-embed --data-dir data
```

`pubmed-embed` calls Ollama’s [`POST /api/embed`](https://docs.ollama.com/api/embed) with `input` and `truncate` (not the deprecated `/api/embeddings` + `prompt` path). Older Ollama builds that only expose `/api/embeddings` are tried if `/api/embed` returns 404.

### TEI HTTP (Text Embeddings Inference)

1. Run one TEI process per GPU (recommended for throughput), each bound to a GPU and listening on its own port, e.g. three containers with `CUDA_VISIBLE_DEVICES=0` / `1` / `2` and ports `8080` / `8081` / `8082`.
2. Set `EMBEDDING_SOURCE=tei-http`, `EMBEDDING_MODEL` to match what TEI serves (for paths/metadata), and **`TEI_BASE_URLS=http://127.0.0.1:8080,http://127.0.0.1:8081,http://127.0.0.1:8082`** for a **three-worker** coordinator (`--workers 3`). Each worker is a **CPU-only** HTTP client; shard `k` uses `TEI_BASE_URLS[k % n]`. Alternatively, a **single** `TEI_BASE_URL` with a reverse proxy load balancer is fine.
3. Batching uses the same env as local: `LOCAL_EMBED_BATCH_SIZE` / `TEI_BATCH_SIZE` / `--local-batch-size`.

```bash
EMBEDDING_SOURCE=tei-http TEI_BASE_URL=http://127.0.0.1:8080 uv run pubmed-embed --data-dir data
```

**Spike / benchmark** (fully in-process — no port, no `text-embeddings-router`, no Docker, no sudo):

Compares **Sentence-Transformers** `encode` (same idea as `pubmed-embed` local) with an optional **raw Hugging Face `transformers`** forward + mean pooling (throughput only; pooling may differ from ST). Default runs both back-to-back with memory cleanup between loads.

```bash
uv run pubmed-embed-benchmark --model BAAI/bge-large-en-v1.5
uv run pubmed-embed-benchmark --backends st --model sentence-transformers/all-MiniLM-L6-v2
uv run pubmed-embed-benchmark --backends transformers --batches 5
```

### Local (Sentence-Transformers)

1. Ensure dependencies are installed (`uv sync`). On Linux and Windows, `pyproject.toml` pulls **PyTorch CUDA 12.8** wheels (`+cu128`) from the official PyTorch index; macOS uses the CPU index. Re-run `uv sync` after pulling changes so the lockfile matches.
2. Set `EMBEDDING_SOURCE=local` and `EMBEDDING_MODEL` to a Hugging Face sentence-embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`). The first run may download weights into the Hugging Face cache.
3. Optionally tune `LOCAL_EMBED_BATCH_SIZE` or `--local-batch-size` for throughput vs memory.
4. **GPU:** By default (`LOCAL_EMBED_DEVICE=auto` or unset), the process uses **CUDA** when `torch.cuda.is_available()`, otherwise **MPS** on Apple Silicon, otherwise **CPU**. Set `LOCAL_EMBED_DEVICE=cuda` (or pass `--local-device cuda`) to require a GPU; if PyTorch cannot see CUDA (driver/build mismatch), fix your NVIDIA driver or install a matching PyTorch wheel from [pytorch.org](https://pytorch.org).

```bash
uv run pubmed-embed --data-dir data
```

Embeddings are computed in the same Python process; vectors are L2-normalized before indexing (same as the Ollama path).

Output layout (example for `bge-m3`):

- `data/embeddings/bge-m3/vectors.faiss` — L2-normalized vectors in `IndexIDMap2` over `IndexFlatIP` (cosine via inner product)
- `data/embeddings/bge-m3/state.sqlite` — `embedded_pmids` for resume; PMIDs are also reconciled from the FAISS id map on startup

**Second model** (e.g. `bge-large-en-v1.5`): change `EMBEDDING_MODEL` or pass `--model bge-large-en-v1.5` so output goes under `data/embeddings/bge-large-en-v1.5/` (separate index and state). That short name is resolved to the Hugging Face repo `BAAI/bge-large-en-v1.5` for download; you can also set the full id explicitly.

**Checkpointing:** every `--checkpoint-every` PMIDs (or `EMBEDDING_CHECKPOINT_EVERY`), the **entire** FAISS index is written atomically (`*.tmp` then replace) and PMIDs are recorded in `state.sqlite`. Default is **5000** for **local** runs and **50** for **Ollama** when neither env nor CLI is set—large local jobs should use **thousands–tens of thousands** to avoid spending most time on disk once indices are huge. Each checkpoint logs wall time to stderr (e.g. `Checkpoint: ntotal=… vectors, 12.34s`).

**Stop and resume:** Ctrl+C sets a stop flag; after the current request, a final checkpoint runs. Re-run the same command to continue; already embedded PMIDs are skipped.

**Testing:** `uv run pubmed-embed --limit 10` processes only the first 10 eligible rows.

### Multi-GPU (`local` or `tei-http`)

Use **multiple processes** to embed disjoint PMID subsets in parallel. Partition rule: **`pmid % num_workers == shard_id`**.

#### `EMBEDDING_SOURCE=local`

- **Coordinator:** `uv run pubmed-embed --data-dir data --workers 3` (or `EMBED_NUM_WORKERS=3`). Spawns three subprocesses with **`CUDA_VISIBLE_DEVICES`** set to GPUs `0,1,2` by default.
- **GPU selection:** `--gpu-ids 0,1,2` must list exactly as many ids as `--workers`.

#### `EMBEDDING_SOURCE=tei-http`

- **Coordinator:** same `--workers 3`, but subprocesses do **not** require CUDA in Python—each worker calls **one TEI base URL** from `TEI_BASE_URLS` (or round-robin a single URL). Run **three TEI servers** (one GPU each) and list all three URLs in `TEI_BASE_URLS`.

#### Common

- **Shard outputs:** `data/embeddings/<slug>/shards/0/`, `shards/1/`, … each with its own `vectors.faiss` and `state.sqlite`.
- **Merge:** By default, when all workers exit successfully, the tool **merges** shards into the canonical `data/embeddings/<slug>/vectors.faiss` and `state.sqlite`. Use **`--no-auto-merge`** to skip that step and merge later with:
  `uv run pubmed-embed --merge-shards --data-dir data` (same `EMBEDDING_MODEL` / `--model` and matching `EMBEDDING_SOURCE`).
- **Resume:** Shards skip PMIDs already in their shard DB or in the **merged** canonical `state.sqlite` (if present). After a successful merge, re-embedding those PMIDs requires clearing state or using a fresh output directory.
- **Advanced:** `--shard-id` / `--num-shards` are used by spawned workers; run the coordinator with `--workers` only (not combined with `--shard-id` on the same command).

Downstream search should use the **merged** canonical index under `data/embeddings/<slug>/`, not individual shard folders.

**Throughput / memory:** For **local**, three workers each load a full copy of the model—raise `TEI_BATCH_SIZE` / `LOCAL_EMBED_BATCH_SIZE` gradually while watching **GPU memory** and **system RAM**. If workers die with exit code **-9** (SIGKILL), the OS may have **OOM-killed** the process; lower batch size or worker count. Prefer a **large** `EMBEDDING_CHECKPOINT_EVERY` (e.g. **10000–50000** for long multi-shard runs) so checkpoints stay fast as shard indices grow.

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
