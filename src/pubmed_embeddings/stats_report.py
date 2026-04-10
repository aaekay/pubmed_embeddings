from __future__ import annotations

import argparse
import pathlib
import re
import sqlite3
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _is_missing(text: str | None) -> bool:
    if text is None:
        return True
    return len(text.strip()) == 0


def _word_count(text: str | None) -> int:
    if not text or not text.strip():
        return 0
    return len(re.split(r"\s+", text.strip()))


def _combined_word_count(title: str | None, abstract: str | None) -> int:
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if abstract and abstract.strip():
        parts.append(abstract.strip())
    if not parts:
        return 0
    return len(re.split(r"\s+", " ".join(parts)))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize PubMed articles SQLite DB and write charts to a folder."
    )
    p.add_argument(
        "--db",
        type=pathlib.Path,
        default=None,
        help="SQLite DB with articles table (default: <data-dir>/pubmed.sqlite)",
    )
    p.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Project data directory (default: data)",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for PNGs and summary (default: <data-dir>/stats_report)",
    )
    p.add_argument(
        "--word-bins",
        type=int,
        default=30,
        help="Number of histogram bins for combined title+abstract word counts (default: 30)",
    )
    p.add_argument(
        "--word-max",
        type=int,
        default=None,
        help="Clip combined word count at this value for histogram x-axis (default: 99.5th percentile, min 500)",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = (args.db if args.db is not None else data_dir / "pubmed.sqlite").resolve()
    out_dir = (
        args.out_dir if args.out_dir is not None else data_dir / "stats_report"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute("SELECT pmid, title, abstract, year FROM articles"))
    finally:
        conn.close()

    n = len(rows)
    title_missing = sum(1 for r in rows if _is_missing(r["title"]))
    abstract_missing = sum(1 for r in rows if _is_missing(r["abstract"]))
    both_missing = sum(
        1 for r in rows if _is_missing(r["title"]) and _is_missing(r["abstract"])
    )
    both_present = sum(
        1
        for r in rows
        if not _is_missing(r["title"]) and not _is_missing(r["abstract"])
    )

    year_counts: dict[str, int] = {}
    for r in rows:
        y = r["year"]
        key = str(int(y)) if y is not None else "unknown"
        year_counts[key] = year_counts.get(key, 0) + 1

    combined_words = np.array(
        [_combined_word_count(r["title"], r["abstract"]) for r in rows], dtype=np.float64
    )

    lines = [
        f"Total articles: {n}",
        f"Title missing: {title_missing}",
        f"Abstract missing: {abstract_missing}",
        f"Both title and abstract missing: {both_missing}",
        f"Both title and abstract present: {both_present}",
        "",
        "Articles per year:",
    ]
    for y in sorted(year_counts.keys(), key=lambda k: (k == "unknown", k)):
        lines.append(f"  {y}: {year_counts[y]}")
    lines.append("")
    lines.append(
        f"Combined word count (title + abstract, whitespace tokens): "
        f"min={int(combined_words.min()) if n else 0}, "
        f"max={int(combined_words.max()) if n else 0}, "
        f"mean={float(combined_words.mean()) if n else 0:.2f}"
    )

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- figures ---
    plt.style.use("default")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    years_sorted = sorted(
        [k for k in year_counts if k != "unknown"],
        key=lambda x: int(x),
    )
    if "unknown" in year_counts:
        years_sorted.append("unknown")
    counts_y = [year_counts[y] for y in years_sorted]
    ax1.bar(range(len(years_sorted)), counts_y, color="steelblue", edgecolor="white")
    ax1.set_xticks(range(len(years_sorted)))
    ax1.set_xticklabels(years_sorted, rotation=45, ha="right")
    ax1.set_xlabel("Publication year")
    ax1.set_ylabel("Number of articles")
    ax1.set_title("Total articles by year")
    fig1.tight_layout()
    fig1.savefig(out_dir / "articles_per_year.png", dpi=150)
    plt.close(fig1)

    if n > 0:
        wmax = args.word_max
        if wmax is None:
            p995 = float(np.percentile(combined_words, 99.5))
            wmax = max(500, int(np.ceil(p995 / 50) * 50))
        wmax = max(1, wmax)
        clipped = np.minimum(combined_words, wmax)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(
            clipped,
            bins=min(args.word_bins, max(5, n)),
            range=(0, wmax),
            color="seagreen",
            edgecolor="white",
        )
        ax2.set_xlabel(f"Combined word count (title + abstract), capped at {wmax}")
        ax2.set_ylabel("Number of articles")
        ax2.set_title("Distribution of combined word counts (histogram)")
        fig2.tight_layout()
        fig2.savefig(out_dir / "word_count_histogram.png", dpi=150)
        plt.close(fig2)
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.text(0.5, 0.5, "No articles in database", ha="center", va="center")
        ax2.set_axis_off()
        fig2.savefig(out_dir / "word_count_histogram.png", dpi=150)
        plt.close(fig2)

    print(f"Wrote stats to {out_dir}")
    print(f"  {summary_path.name}")
    print("  articles_per_year.png")
    print("  word_count_histogram.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
