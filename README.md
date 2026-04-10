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
