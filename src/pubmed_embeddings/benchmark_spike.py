"""In-process embedding throughput benchmark — no port, no router, no Docker, no sudo.

Runs entirely in Python / PyTorch:

1. **Sentence-Transformers** ``encode`` (typical ``pubmed-embed`` path).
2. Optional **raw Hugging Face ``transformers``** forward + mean pooling + L2 normalize
   (rough throughput comparison; pooling may differ from ST for some architectures).

Both use the same Hugging Face model id. Models are loaded **one after the other** with
cleanup between runs to reduce peak VRAM when comparing both.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time

from pubmed_embeddings.embeddings import _resolve_local_sentence_transformer_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark in-process embedding throughput (Sentence-Transformers and/or raw transformers). "
            "No server, port, Docker, or sudo."
        )
    )
    p.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Hugging Face model id (default: BAAI/bge-large-en-v1.5)",
    )
    p.add_argument(
        "--backends",
        type=str,
        default="st,transformers",
        help="Comma-separated: st, transformers (default: st,transformers)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=20, help="Timed batches after warmup")
    p.add_argument("--warmup", type=int, default=3, help="Warmup batches (not timed)")
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length for raw transformers path (default 512)",
    )
    return p.parse_args()


def _mean_pool(last_hidden: object, attention_mask: object) -> object:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _run_sentence_transformers(
    texts: list[str],
    *,
    mid: str,
    device: str,
    batch_size: int,
    warmup: int,
    batches: int,
) -> tuple[list[float], float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(mid, device=device)

    def once() -> None:
        model.encode(
            texts,
            batch_size=min(batch_size, len(texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    for _ in range(warmup):
        once()

    times: list[float] = []
    for _ in range(batches):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    total = sum(times)
    return times, total


def _run_transformers_raw(
    texts: list[str],
    *,
    mid: str,
    device: str,
    batch_size: int,
    warmup: int,
    batches: int,
    max_length: int,
) -> tuple[list[float], float]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid)
    model = model.to(device)
    model.eval()

    def once() -> None:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                batch = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                emb = _mean_pool(out.last_hidden_state, batch["attention_mask"])
                _ = torch.nn.functional.normalize(emb, p=2, dim=1)

    for _ in range(warmup):
        once()

    times: list[float] = []
    for _ in range(batches):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    total = sum(times)
    return times, total


def _release_torch_memory(device: str) -> None:
    import torch

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> int:
    args = _parse_args()
    want = {x.strip().lower() for x in args.backends.split(",") if x.strip()}
    if not want:
        print("--backends must list at least one of: st, transformers", file=sys.stderr)
        return 2
    unknown = want - {"st", "transformers"}
    if unknown:
        print(f"Unknown backend(s): {unknown}", file=sys.stderr)
        return 2

    texts = [
        "Deep learning for biomedical text retrieval " + f"token {i % 100}"
        for i in range(args.batch_size)
    ]
    mid = _resolve_local_sentence_transformer_model(args.model)

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available; benchmarks use CPU.", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n = args.batch_size * args.batches
    st_total: float | None = None
    tr_total: float | None = None

    if "st" in want:
        st_times, st_total = _run_sentence_transformers(
            texts,
            mid=mid,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            batches=args.batches,
        )
        print(
            f"Sentence-Transformers ({device}) model={mid}: "
            f"{n} texts in {st_total:.3f}s => {n / st_total:.1f} texts/s "
            f"(batch={args.batch_size}, batches={args.batches}; "
            f"batch p50={statistics.median(st_times)*1000:.1f}ms)"
        )
        # Drop ST model before loading a second full weights copy.
        del st_times
        _release_torch_memory(device)

    if "transformers" in want:
        tr_times, tr_total = _run_transformers_raw(
            texts,
            mid=mid,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            batches=args.batches,
            max_length=args.max_length,
        )
        print(
            f"transformers mean-pool ({device}) model={mid}: "
            f"{n} texts in {tr_total:.3f}s => {n / tr_total:.1f} texts/s "
            f"(batch={args.batch_size}, batches={args.batches}; max_length={args.max_length}; "
            f"batch p50={statistics.median(tr_times)*1000:.1f}ms)"
        )
        print(
            "Note: raw path uses mean pooling; Sentence-Transformers may use different pooling — "
            "compare throughput only, not vector identity.",
            file=sys.stderr,
        )

    if st_total is not None and tr_total is not None and st_total > 0 and tr_total > 0:
        ratio = (n / tr_total) / (n / st_total)
        print(f"Throughput ratio transformers/ST: {ratio:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
