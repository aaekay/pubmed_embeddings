from __future__ import annotations

import argparse
import gzip
import pathlib
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from typing import Iterable, Iterator

from tqdm import tqdm


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _element_text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    text = "".join(elem.itertext()).strip()
    return text or None


def _find_child(parent: ET.Element, *names: str) -> ET.Element | None:
    for name in names:
        for child in parent:
            if _local_tag(child.tag) == name:
                return child
    return None


def _parse_year_from_medline_date(medline_date: ET.Element | None) -> int | None:
    if medline_date is None:
        return None
    raw = _element_text(medline_date)
    if not raw:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", raw)
    if match:
        return int(match.group(0))
    return None


def _extract_publication_year(medline: ET.Element) -> int | None:
    article = _find_child(medline, "Article")
    if article is None:
        return None

    journal = _find_child(article, "Journal")
    if journal is not None:
        issue = _find_child(journal, "JournalIssue")
        if issue is not None:
            pub_date = _find_child(issue, "PubDate")
            if pub_date is not None:
                year_el = _find_child(pub_date, "Year")
                if year_el is not None and year_el.text:
                    try:
                        return int(year_el.text.strip())
                    except ValueError:
                        pass
                medline_date = _find_child(pub_date, "MedlineDate")
                y = _parse_year_from_medline_date(medline_date)
                if y is not None:
                    return y

    article_date = _find_child(article, "ArticleDate")
    if article_date is not None:
        year_el = _find_child(article_date, "Year")
        if year_el is not None and year_el.text:
            try:
                return int(year_el.text.strip())
            except ValueError:
                pass

    for path in (
        ("Article", "Journal", "JournalIssue", "PubDate"),
        ("Article", "ArticleDate"),
    ):
        el: ET.Element | None = medline
        for part in path:
            el = _find_child(el, part) if el is not None else None
        if el is not None:
            y_el = _find_child(el, "Year")
            if y_el is not None and y_el.text:
                try:
                    return int(y_el.text.strip())
                except ValueError:
                    pass

    date_completed = _find_child(medline, "DateCompleted")
    if date_completed is not None:
        year_el = _find_child(date_completed, "Year")
        if year_el is not None and year_el.text:
            try:
                return int(year_el.text.strip())
            except ValueError:
                pass

    return None


def _extract_abstract(article: ET.Element) -> str | None:
    abstract = _find_child(article, "Abstract")
    if abstract is None:
        return None
    parts: list[str] = []
    for child in abstract:
        if _local_tag(child.tag) != "AbstractText":
            continue
        label = child.attrib.get("Label")
        body = _element_text(child) or ""
        if label:
            parts.append(f"{label}: {body}".strip())
        else:
            parts.append(body.strip())
    merged = "\n\n".join(p for p in parts if p)
    return merged or None


def _extract_from_pubmed_article(root: ET.Element) -> tuple[int, str | None, str | None, int | None, str | None] | None:
    medline = _find_child(root, "MedlineCitation")
    if medline is None:
        return None

    pmid_el = _find_child(medline, "PMID")
    if pmid_el is None:
        return None
    pmid_raw = "".join(pmid_el.itertext()).strip()
    if not pmid_raw:
        return None
    try:
        pmid = int(pmid_raw)
    except ValueError:
        return None

    article = _find_child(medline, "Article")
    if article is None:
        return None

    title_el = _find_child(article, "ArticleTitle")
    title = _element_text(title_el) if title_el is not None else None

    abstract = _extract_abstract(article)

    journal = _find_child(article, "Journal")
    journal_name: str | None = None
    if journal is not None:
        jt = _find_child(journal, "Title")
        journal_name = _element_text(jt) if jt is not None else None

    year = _extract_publication_year(medline)

    return (pmid, title, abstract, year, journal_name)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            pmid INTEGER PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            year INTEGER,
            journal TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_files (
            basename TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (basename, size_bytes, mtime_ns)
        )
        """
    )
    conn.commit()


def _configure_sqlite(conn: sqlite3.Connection, *, fast: bool) -> None:
    """Tune SQLite for bulk ingest. WAL + NORMAL sync is a good default; optional fast mode trades safety for speed."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")
    conn.execute("PRAGMA mmap_size = 268435456")
    if fast:
        conn.execute("PRAGMA synchronous = OFF")
    else:
        conn.execute("PRAGMA synchronous = NORMAL")


def _file_fingerprint(path: pathlib.Path) -> tuple[str, int, int]:
    st = path.stat()
    return path.name, st.st_size, int(st.st_mtime_ns)


def _is_file_already_ingested(
    conn: sqlite3.Connection, fp: tuple[str, int, int]
) -> bool:
    basename, size_bytes, mtime_ns = fp
    row = conn.execute(
        """
        SELECT 1 FROM ingested_files
        WHERE basename = ? AND size_bytes = ? AND mtime_ns = ?
        """,
        (basename, size_bytes, mtime_ns),
    ).fetchone()
    return row is not None


def _mark_file_ingested(conn: sqlite3.Connection, fp: tuple[str, int, int]) -> None:
    basename, size_bytes, mtime_ns = fp
    conn.execute(
        """
        INSERT OR REPLACE INTO ingested_files
            (basename, size_bytes, mtime_ns, ingested_at)
        VALUES (?, ?, ?, ?)
        """,
        (basename, size_bytes, mtime_ns, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )
    conn.commit()


def _iter_pubmed_events(
    path: pathlib.Path,
) -> Iterator[tuple[str, ET.Element]]:
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rb") as raw:
        context = ET.iterparse(raw, events=("end",))
        for event, elem in context:
            tag = _local_tag(elem.tag)
            yield tag, elem


def extract_file(
    path: pathlib.Path,
    conn: sqlite3.Connection,
    batch_size: int,
    stats: dict[str, int],
) -> None:
    batch: list[tuple[int, str | None, str | None, int | None, str | None]] = []
    delete_batch: list[tuple[int]] = []

    def flush_rows() -> None:
        if not batch:
            return
        conn.executemany(
            """
            INSERT INTO articles (pmid, title, abstract, year, journal)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                year = excluded.year,
                journal = excluded.journal
            """,
            batch,
        )
        conn.commit()
        stats["rows"] = stats.get("rows", 0) + len(batch)
        batch.clear()

    def flush_deletes() -> None:
        if not delete_batch:
            return
        conn.executemany("DELETE FROM articles WHERE pmid = ?", delete_batch)
        conn.commit()
        stats["deleted"] = stats.get("deleted", 0) + len(delete_batch)
        delete_batch.clear()

    for tag, elem in _iter_pubmed_events(path):
        if tag == "PubmedArticle":
            row = _extract_from_pubmed_article(elem)
            if row is not None:
                batch.append(row)
                if len(batch) >= batch_size:
                    flush_rows()
            elem.clear()
        elif tag == "DeleteCitation":
            pmid_el = _find_child(elem, "PMID")
            pmid_raw = "".join(pmid_el.itertext()).strip() if pmid_el is not None else ""
            if pmid_raw:
                try:
                    pmid = int(pmid_raw)
                    delete_batch.append((pmid,))
                    if len(delete_batch) >= batch_size:
                        flush_deletes()
                except ValueError:
                    pass
            elem.clear()
    flush_rows()
    flush_deletes()


def _discover_xml_gz(data_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(data_dir.glob("*.xml.gz"))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PMID, title, abstract, year, journal from PubMed XML into SQLite."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=pathlib.Path,
        help="Directory containing pubmed*.xml.gz files (default: data)",
    )
    parser.add_argument(
        "--db",
        default=None,
        type=pathlib.Path,
        help="SQLite database path (default: <data-dir>/pubmed.sqlite)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per transaction (default: 500)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster ingest: PRAGMA synchronous=OFF (slightly higher risk if the OS crashes mid-write)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-ingest every file even if a previous run finished it (same basename+size+mtime)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=pathlib.Path,
        help="Specific XML or XML.GZ files; default: all *.xml.gz in data-dir",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        paths = [pathlib.Path(p).resolve() for p in args.files]
    else:
        paths = _discover_xml_gz(data_dir)

    if not paths:
        print(f"No XML files found under {data_dir} (expected *.xml.gz)")
        return 1

    db_path = (args.db if args.db is not None else data_dir / "pubmed.sqlite").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        _configure_sqlite(conn, fast=args.fast)
        _ensure_schema(conn)
        stats: dict[str, int] = {"rows": 0, "deleted": 0, "skipped_files": 0}

        for xml_path in tqdm(paths, desc="files", unit="file"):
            fp = _file_fingerprint(xml_path)
            if not args.no_resume and _is_file_already_ingested(conn, fp):
                stats["skipped_files"] = stats.get("skipped_files", 0) + 1
                continue
            extract_file(xml_path, conn, batch_size=max(1, args.batch_size), stats=stats)
            _mark_file_ingested(conn, fp)

        print(
            f"Done. Upserted rows (cumulative commits): {stats.get('rows', 0)}; "
            f"deletes from update files: {stats.get('deleted', 0)}; "
            f"files skipped (resume): {stats.get('skipped_files', 0)}. "
            f"Database: {db_path}"
        )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
