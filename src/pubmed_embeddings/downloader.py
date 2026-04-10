from __future__ import annotations

import argparse
import concurrent.futures
import html.parser
import pathlib
import re
import threading
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from tqdm import tqdm


BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
UPDATES_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
FILE_PATTERN = re.compile(r".+\.xml\.gz$")
CHUNK_SIZE = 1024 * 1024


class _LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str


@dataclass(frozen=True)
class DownloadJob:
    remote: RemoteFile
    target_path: pathlib.Path
    expected_size: int | None


@dataclass(frozen=True)
class DownloadResult:
    file_name: str
    status: str
    message: str


def list_remote_files(directory_url: str) -> list[RemoteFile]:
    with urllib.request.urlopen(directory_url) as response:
        html_body = response.read().decode("utf-8", errors="replace")

    parser = _LinkParser()
    parser.feed(html_body)

    names = sorted(
        {
            link
            for link in parser.links
            if FILE_PATTERN.match(link) and not link.startswith("..")
        }
    )
    return [RemoteFile(name=name, url=f"{directory_url}{name}") for name in names]


def remote_size_bytes(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            content_length = response.headers.get("Content-Length")
    except Exception:
        return None

    if not content_length:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def should_download(path: pathlib.Path, remote_url: str, force: bool) -> bool:
    if force or not path.exists():
        return True

    remote_size = remote_size_bytes(remote_url)
    if remote_size is None:
        return False
    return path.stat().st_size != remote_size


def _stream_download(
    remote: RemoteFile,
    target_path: pathlib.Path,
    bytes_bar: tqdm | None,
    bytes_lock: threading.Lock,
) -> None:
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    downloaded = 0

    try:
        with urllib.request.urlopen(remote.url) as response, tmp_path.open("wb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                chunk_len = len(chunk)
                downloaded += chunk_len
                if bytes_bar is not None:
                    with bytes_lock:
                        bytes_bar.update(chunk_len)
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _build_jobs(
    remote_files: list[RemoteFile],
    target_dir: pathlib.Path,
    force: bool,
) -> tuple[list[DownloadJob], int]:
    jobs: list[DownloadJob] = []
    skipped = 0

    for remote in remote_files:
        target_path = target_dir / remote.name
        if force or not target_path.exists():
            jobs.append(DownloadJob(remote=remote, target_path=target_path, expected_size=None))
            continue

        expected_size = remote_size_bytes(remote.url)
        if expected_size is None:
            skipped += 1
            continue

        if target_path.stat().st_size == expected_size:
            skipped += 1
            continue

        jobs.append(
            DownloadJob(
                remote=remote,
                target_path=target_path,
                expected_size=expected_size,
            )
        )
    return jobs, skipped


def download_group(
    label: str,
    source_url: str,
    target_dir: pathlib.Path,
    limit: int | None,
    force: bool,
    workers: int,
) -> None:
    print(f"[{label}] listing files from {source_url}")
    remote_files = list_remote_files(source_url)
    if limit is not None:
        remote_files = remote_files[:limit]

    if not remote_files:
        print(f"[{label}] no files found")
        return

    jobs, skipped = _build_jobs(remote_files, target_dir=target_dir, force=force)
    total = len(remote_files)
    completed = skipped
    saved = 0
    failed = 0

    bytes_total = sum(job.expected_size for job in jobs if job.expected_size is not None)
    bytes_bar_total = bytes_total if bytes_total > 0 else None
    bytes_lock = threading.Lock()

    with tqdm(total=total, desc=f"{label} files", unit="file") as files_bar:
        if skipped:
            files_bar.update(skipped)
            tqdm.write(f"[{label}] skipped {skipped} unchanged file(s)")

        bytes_bar: tqdm | None
        if jobs:
            bytes_bar = tqdm(
                total=bytes_bar_total,
                desc=f"{label} bytes",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            )
        else:
            bytes_bar = None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        _stream_download,
                        job.remote,
                        job.target_path,
                        bytes_bar,
                        bytes_lock,
                    ): job
                    for job in jobs
                }

                for future in concurrent.futures.as_completed(future_map):
                    job = future_map[future]
                    try:
                        future.result()
                        saved += 1
                        completed += 1
                        result = DownloadResult(
                            file_name=job.remote.name,
                            status="saved",
                            message=f"[{label}] saved {job.remote.name}",
                        )
                    except Exception as exc:
                        failed += 1
                        completed += 1
                        result = DownloadResult(
                            file_name=job.remote.name,
                            status="failed",
                            message=f"[{label}] failed {job.remote.name}: {exc}",
                        )

                    files_bar.update(1)
                    tqdm.write(result.message)
        finally:
            if bytes_bar is not None:
                bytes_bar.close()

    print(
        f"[{label}] complete: checked={total}, saved={saved}, skipped={skipped}, failed={failed}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PubMed baseline and/or update XML archives."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Local output directory (default: data)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Download annual baseline files",
    )
    parser.add_argument(
        "--updates",
        action="store_true",
        help="Download daily update files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files per source (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    target_dir = pathlib.Path(args.data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    run_baseline = args.baseline
    run_updates = args.updates
    if not run_baseline and not run_updates:
        run_baseline = True
        run_updates = True

    if run_baseline:
        download_group(
            label="baseline",
            source_url=BASELINE_URL,
            target_dir=target_dir,
            limit=args.limit,
            force=args.force,
            workers=max(1, args.workers),
        )
    if run_updates:
        download_group(
            label="updates",
            source_url=UPDATES_URL,
            target_dir=target_dir,
            limit=args.limit,
            force=args.force,
            workers=max(1, args.workers),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
