from __future__ import annotations

import argparse
import html.parser
import pathlib
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterable


BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
UPDATES_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
FILE_PATTERN = re.compile(r".+\.xml\.gz$")


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


def download_file(remote: RemoteFile, target_dir: pathlib.Path, force: bool) -> str:
    target_path = target_dir / remote.name
    if not should_download(target_path, remote.url, force=force):
        return f"skip {remote.name}"

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    urllib.request.urlretrieve(remote.url, tmp_path)
    tmp_path.replace(target_path)
    return f"saved {remote.name}"


def download_group(
    label: str,
    source_url: str,
    target_dir: pathlib.Path,
    limit: int | None,
    force: bool,
) -> None:
    print(f"[{label}] listing files from {source_url}")
    remote_files = list_remote_files(source_url)
    if limit is not None:
        remote_files = remote_files[:limit]

    if not remote_files:
        print(f"[{label}] no files found")
        return

    for remote in remote_files:
        status = download_file(remote, target_dir=target_dir, force=force)
        print(f"[{label}] {status}")

    print(f"[{label}] complete ({len(remote_files)} files checked)")


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
        )
    if run_updates:
        download_group(
            label="updates",
            source_url=UPDATES_URL,
            target_dir=target_dir,
            limit=args.limit,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
