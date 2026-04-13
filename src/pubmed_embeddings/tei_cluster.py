from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, TextIO

import httpx


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS = (11450, 11451, 11452)
DEFAULT_GPU_IDS = (0, 1, 2)
DEFAULT_TEI_VERSION = "v1.8.3"
TEI_SOURCE_URL = (
    "https://github.com/huggingface/text-embeddings-inference/archive/refs/tags/{version}.tar.gz"
)
REQUIRED_ROUTER_FLAGS = (
    "--model-id",
    "--hostname",
    "--port",
    "--uds-path",
    "--huggingface-hub-cache",
)
STOP_EVENT = threading.Event()


@dataclass(frozen=True)
class ManagedPaths:
    cache_root: pathlib.Path
    cargo_home: pathlib.Path
    rustup_home: pathlib.Path
    install_root: pathlib.Path
    source_root: pathlib.Path
    downloads_root: pathlib.Path
    logs_root: pathlib.Path
    hub_cache: pathlib.Path


@dataclass(frozen=True)
class Toolchain:
    cargo: pathlib.Path
    env_overrides: dict[str, str]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    compute_cap: float


@dataclass
class LaunchedProcess:
    gpu_id: int
    port: int
    base_url: str
    log_path: pathlib.Path
    process: subprocess.Popen[str]
    output_thread: threading.Thread


def _handle_stop_signal(_signum: int, _frame: object | None) -> None:
    STOP_EVENT.set()


def _default_cache_root() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "pubmed_embeddings" / "tei"


def _managed_paths(cache_root: pathlib.Path) -> ManagedPaths:
    rust_root = cache_root / "rust"
    return ManagedPaths(
        cache_root=cache_root,
        cargo_home=rust_root / "cargo",
        rustup_home=rust_root / "rustup",
        install_root=cache_root / "install",
        source_root=cache_root / "src",
        downloads_root=cache_root / "downloads",
        logs_root=cache_root / "logs",
        hub_cache=cache_root / "huggingface",
    )


def _source_dir(paths: ManagedPaths, version: str) -> pathlib.Path:
    suffix = version[1:] if version.startswith("v") else version
    return paths.source_root / f"text-embeddings-inference-{suffix}"


def _install_meta_path(paths: ManagedPaths) -> pathlib.Path:
    return paths.install_root / "install-meta.json"


def _managed_router_binary(paths: ManagedPaths) -> pathlib.Path:
    return paths.install_root / "bin" / "text-embeddings-router"


def _parse_csv_ints(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            raise ValueError(f"{name} contains an empty item: {raw!r}")
        try:
            values.append(int(piece))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of integers: {raw!r}") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _router_base_urls(host: str, ports: list[int]) -> list[str]:
    return [f"http://{host}:{port}" for port in ports]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reuse a compatible TEI router on PATH or build TEI in a user-space cache, "
            "then launch one TEI server per GPU "
            "and keep the cluster in the foreground."
        )
    )
    p.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id to serve from all TEI workers.",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face model revision for all TEI workers.",
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind host for all TEI workers (default: {DEFAULT_HOST}).",
    )
    p.add_argument(
        "--ports",
        default=",".join(str(x) for x in DEFAULT_PORTS),
        help="Comma-separated TEI ports (default: 11450,11451,11452).",
    )
    p.add_argument(
        "--gpu-ids",
        default=",".join(str(x) for x in DEFAULT_GPU_IDS),
        help="Comma-separated physical GPU ids for TEI workers (default: 0,1,2).",
    )
    p.add_argument(
        "--tei-version",
        default=DEFAULT_TEI_VERSION,
        help=f"TEI git tag to install when managed install is needed (default: {DEFAULT_TEI_VERSION}).",
    )
    p.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help="Optional TEI --max-batch-tokens value for all workers.",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token; defaults to HF_TOKEN from the environment.",
    )
    p.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        help="Cache/install root (default: ~/.cache/pubmed_embeddings/tei).",
    )
    p.add_argument(
        "--startup-timeout",
        type=int,
        default=900,
        help="Seconds to wait for each worker to become healthy (default: 900).",
    )
    return p.parse_args(argv)


def _query_visible_gpus() -> list[GpuInfo]:
    if sys.platform != "linux":
        raise RuntimeError(
            "pubmed-tei-cluster currently targets Linux NVIDIA servers only."
        )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError(
            "nvidia-smi was not found on PATH. A TEI CUDA cluster requires NVIDIA tooling on the server."
        )

    cmd = [
        nvidia_smi,
        "--query-gpu=index,name,compute_cap",
        "--format=csv,noheader",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi failed while querying GPUs: {details}")

    gpus: list[GpuInfo] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise RuntimeError(f"Could not parse nvidia-smi output line: {line!r}")
        try:
            idx = int(parts[0])
            compute_cap = float(parts[2])
        except ValueError as exc:
            raise RuntimeError(f"Could not parse nvidia-smi output line: {line!r}") from exc
        gpus.append(GpuInfo(index=idx, name=parts[1], compute_cap=compute_cap))

    if not gpus:
        raise RuntimeError("No NVIDIA GPUs were reported by nvidia-smi.")
    return gpus


def _select_gpus(all_gpus: list[GpuInfo], gpu_ids: list[int]) -> list[GpuInfo]:
    by_id = {gpu.index: gpu for gpu in all_gpus}
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in by_id]
    if missing:
        known = ",".join(str(gpu.index) for gpu in all_gpus)
        raise RuntimeError(
            f"Requested GPU ids {missing} are not visible. Visible GPU ids: {known or 'none'}."
        )
    return [by_id[gpu_id] for gpu_id in gpu_ids]


def _cuda_feature_for_gpus(gpus: list[GpuInfo]) -> str:
    families: set[str] = set()
    for gpu in gpus:
        if gpu.compute_cap < 7.5:
            raise RuntimeError(
                f"GPU {gpu.index} ({gpu.name}) has compute capability {gpu.compute_cap}, "
                "but TEI CUDA builds require 7.5 or newer."
            )
        if gpu.compute_cap < 8.0:
            families.add("turing")
        else:
            families.add("modern")
    if len(families) > 1:
        raise RuntimeError(
            "The selected GPUs mix Turing-class and Ampere-or-newer devices. "
            "The managed TEI install expects a homogeneous GPU family."
        )
    return "candle-cuda-turing" if "turing" in families else "candle-cuda"


def _prepend_path(env: dict[str, str], candidate: pathlib.Path) -> None:
    current = env.get("PATH", "")
    prefix = str(candidate)
    env["PATH"] = prefix if not current else f"{prefix}:{current}"


def _normalize_version_token(raw: str) -> str:
    return raw.strip().lower().lstrip("v")


def _extract_semver(text: str) -> str | None:
    match = re.search(r"\bv?(\d+\.\d+\.\d+)\b", text)
    if match:
        return match.group(1)
    return None


def _probe_router_binary(
    binary: pathlib.Path,
    *,
    expected_version: str,
    require_max_batch_tokens: bool,
    require_revision: bool,
) -> str:
    help_proc = subprocess.run(
        [str(binary), "--help"],
        capture_output=True,
        text=True,
    )
    help_text = ((help_proc.stdout or "") + "\n" + (help_proc.stderr or "")).strip()
    if help_proc.returncode != 0:
        raise RuntimeError(
            f"{binary} --help failed with exit code {help_proc.returncode}: {help_text[:4000]}"
        )

    required_flags = list(REQUIRED_ROUTER_FLAGS)
    if require_max_batch_tokens:
        required_flags.append("--max-batch-tokens")
    if require_revision:
        required_flags.append("--revision")
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        raise RuntimeError(
            f"{binary} is missing required CLI flags: {', '.join(missing)}"
        )

    version_proc = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
    )
    version_text = ((version_proc.stdout or "") + "\n" + (version_proc.stderr or "")).strip()
    if version_proc.returncode != 0:
        raise RuntimeError(
            f"{binary} --version failed with exit code {version_proc.returncode}: {version_text[:4000]}"
        )

    detected = _extract_semver(version_text)
    expected = _normalize_version_token(expected_version)
    if detected is None:
        raise RuntimeError(
            f"{binary} did not report a parseable semantic version: {version_text[:4000]}"
        )
    if detected != expected:
        raise RuntimeError(
            f"{binary} reports TEI version {detected}, expected {expected}"
        )
    return detected


def _ensure_cuda_toolkit_on_path(env: dict[str, str]) -> None:
    candidates: list[pathlib.Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = env.get(key)
        if value:
            candidates.append(pathlib.Path(value) / "bin")
    candidates.append(pathlib.Path("/usr/local/cuda/bin"))

    for candidate in candidates:
        if candidate.exists():
            _prepend_path(env, candidate)

    if shutil.which("nvcc", path=env.get("PATH")) is None:
        raise RuntimeError(
            "CUDA toolkit binaries were not found. TEI local CUDA builds require nvcc on PATH "
            "(for example under /usr/local/cuda/bin)."
        )


def _download_to(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _validate_tei_archive(archive_path: pathlib.Path, *, expected_root: str) -> None:
    required_member = f"{expected_root}/router/Cargo.toml"
    with tarfile.open(archive_path, mode="r:gz") as tf:
        members = [member.name for member in tf.getmembers()]
    if not members:
        raise RuntimeError(f"Downloaded TEI archive is empty: {archive_path}")
    if required_member not in members:
        raise RuntimeError(
            f"Downloaded TEI archive is missing {required_member}; refusing to build from it."
        )
    bad_roots = [
        member
        for member in members
        if member not in (expected_root, ".")
        and not member.startswith(f"{expected_root}/")
    ]
    if bad_roots:
        raise RuntimeError(
            f"Downloaded TEI archive has unexpected top-level paths, first bad path: {bad_roots[0]!r}"
        )


def _safe_extract_tar_gz(archive_path: pathlib.Path, dest_dir: pathlib.Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            target = (dest_dir / member.name).resolve()
            if os.path.commonpath([str(dest_root), str(target)]) != str(dest_root):
                raise RuntimeError(f"Unsafe archive path while extracting {archive_path.name}: {member.name}")
        tf.extractall(dest_dir)


def _resolve_toolchain(paths: ManagedPaths) -> Toolchain:
    cargo_on_path = shutil.which("cargo")
    if cargo_on_path is not None:
        return Toolchain(cargo=pathlib.Path(cargo_on_path), env_overrides={})

    managed_cargo = paths.cargo_home / "bin" / "cargo"
    if managed_cargo.exists():
        env = os.environ.copy()
        env["CARGO_HOME"] = str(paths.cargo_home)
        env["RUSTUP_HOME"] = str(paths.rustup_home)
        _prepend_path(env, managed_cargo.parent)
        return Toolchain(
            cargo=managed_cargo,
            env_overrides={
                "CARGO_HOME": str(paths.cargo_home),
                "RUSTUP_HOME": str(paths.rustup_home),
                "PATH": env["PATH"],
            },
        )

    raise RuntimeError(
        "cargo was not found on PATH. Install a Rust toolchain before running pubmed-tei-cluster. "
        "Automatic rustup installation has been removed for safety."
    )


def _load_install_meta(paths: ManagedPaths) -> dict[str, str]:
    meta_path = _install_meta_path(paths)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_install_meta(paths: ManagedPaths, *, version: str, feature: str) -> None:
    meta_path = _install_meta_path(paths)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"tei_version": version, "cuda_feature": feature}, indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_tei_source(paths: ManagedPaths, version: str) -> pathlib.Path:
    source_dir = _source_dir(paths, version)
    if source_dir.exists():
        return source_dir

    archive_name = f"{version}.tar.gz"
    archive_path = paths.downloads_root / archive_name
    if not archive_path.exists():
        url = TEI_SOURCE_URL.format(version=version)
        print(f"Downloading TEI source {version} ...", file=sys.stderr)
        try:
            _download_to(url, archive_path)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not download TEI source from {url}: {exc}") from exc

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="tei-src-", dir=paths.source_root))
    try:
        _validate_tei_archive(archive_path, expected_root=source_dir.name)
        _safe_extract_tar_gz(archive_path, tmp_dir)
        extracted = tmp_dir / source_dir.name
        if not extracted.exists():
            raise RuntimeError(
                f"Downloaded TEI archive did not contain expected folder {source_dir.name}."
            )
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(source_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return source_dir


def _ensure_build_prerequisites(env: dict[str, str]) -> None:
    if shutil.which("gcc", path=env.get("PATH")) is None:
        raise RuntimeError(
            "gcc was not found on PATH. TEI local builds require a working GCC toolchain."
        )
    _ensure_cuda_toolkit_on_path(env)


def _ensure_managed_router_binary(
    paths: ManagedPaths,
    *,
    version: str,
    feature: str,
    require_max_batch_tokens: bool,
    require_revision: bool,
) -> pathlib.Path:
    binary = _managed_router_binary(paths)
    meta = _load_install_meta(paths)
    if (
        binary.exists()
        and meta.get("tei_version") == version
        and meta.get("cuda_feature") == feature
    ):
        try:
            detected = _probe_router_binary(
                binary,
                expected_version=version,
                require_max_batch_tokens=require_max_batch_tokens,
                require_revision=require_revision,
            )
            print(
                f"Using cached managed text-embeddings-router {detected} at {binary}",
                file=sys.stderr,
            )
            return binary
        except RuntimeError as exc:
            print(
                f"Cached managed text-embeddings-router is incompatible; rebuilding: {exc}",
                file=sys.stderr,
            )

    toolchain = _resolve_toolchain(paths)
    env = os.environ.copy()
    env.update(toolchain.env_overrides)
    _ensure_build_prerequisites(env)

    source_dir = _ensure_tei_source(paths, version)
    router_dir = source_dir / "router"
    if not router_dir.is_dir():
        raise RuntimeError(f"TEI router directory not found: {router_dir}")

    paths.install_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(toolchain.cargo),
        "install",
        "--locked",
        "--path",
        str(router_dir),
        "--root",
        str(paths.install_root),
        "--features",
        feature,
        "--force",
    ]
    print(
        f"Building text-embeddings-router {version} with feature {feature} ...",
        file=sys.stderr,
    )
    try:
        subprocess.run(cmd, cwd=source_dir, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "TEI build failed. Check the build output above. Common causes on no-sudo servers are "
            "missing CUDA toolkit binaries, GCC, or OpenSSL/linker development packages."
        ) from exc

    if not binary.exists():
        raise RuntimeError(f"TEI build finished without producing {binary}.")
    _write_install_meta(paths, version=version, feature=feature)
    detected = _probe_router_binary(
        binary,
        expected_version=version,
        require_max_batch_tokens=require_max_batch_tokens,
        require_revision=require_revision,
    )
    print(
        f"Built managed text-embeddings-router {detected} at {binary}",
        file=sys.stderr,
    )
    return binary


def _resolve_router_binary(
    paths: ManagedPaths,
    *,
    version: str,
    feature: str,
    require_max_batch_tokens: bool,
    require_revision: bool,
) -> pathlib.Path:
    existing = shutil.which("text-embeddings-router")
    if existing is not None:
        existing_path = pathlib.Path(existing)
        try:
            detected = _probe_router_binary(
                existing_path,
                expected_version=version,
                require_max_batch_tokens=require_max_batch_tokens,
                require_revision=require_revision,
            )
            print(
                f"Using compatible text-embeddings-router {detected} from PATH: {existing_path}",
                file=sys.stderr,
            )
            return existing_path
        except RuntimeError as exc:
            print(
                f"Ignoring PATH text-embeddings-router at {existing_path}: {exc}",
                file=sys.stderr,
            )
    return _ensure_managed_router_binary(
        paths,
        version=version,
        feature=feature,
        require_max_batch_tokens=require_max_batch_tokens,
        require_revision=require_revision,
    )


def _stream_output(prefix: str, pipe: TextIO | None, log_path: pathlib.Path) -> None:
    if pipe is None:
        return
    with log_path.open("a", encoding="utf-8") as log_file:
        for raw_line in pipe:
            line = raw_line.rstrip()
            print(f"[{prefix}] {line}", flush=True)
            log_file.write(raw_line)
            log_file.flush()


def _health_check(client: httpx.Client, base_url: str) -> tuple[bool, str]:
    for path in ("/health", "/docs"):
        try:
            response = client.get(f"{base_url}{path}")
        except httpx.HTTPError as exc:
            return False, str(exc)
        if response.is_success:
            return True, path
        if response.status_code not in (404, 405):
            return False, f"{path} -> HTTP {response.status_code}"
    return False, "health endpoint not ready yet"


def _wait_until_ready(instance: LaunchedProcess, *, timeout_s: int) -> None:
    deadline = time.monotonic() + max(1, timeout_s)
    last_error = "no response yet"
    with httpx.Client(timeout=httpx.Timeout(2.0)) as client:
        while time.monotonic() < deadline:
            code = instance.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"TEI worker on GPU {instance.gpu_id} port {instance.port} exited with code {code}. "
                    f"See {instance.log_path}."
                )
            ok, detail = _health_check(client, instance.base_url)
            if ok:
                print(
                    f"TEI worker ready on GPU {instance.gpu_id} -> {instance.base_url} ({detail})",
                    file=sys.stderr,
                )
                return
            last_error = detail
            time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for TEI worker on GPU {instance.gpu_id} port {instance.port}. "
        f"Last check: {last_error}. See {instance.log_path}."
    )


def _build_router_command(
    router_binary: pathlib.Path,
    *,
    model: str,
    revision: str | None,
    host: str,
    port: int,
    hub_cache: pathlib.Path,
    uds_path: pathlib.Path,
    max_batch_tokens: int | None,
) -> list[str]:
    cmd = [
        str(router_binary),
        "--model-id",
        model,
        "--hostname",
        host,
        "--port",
        str(port),
        "--uds-path",
        str(uds_path),
        "--huggingface-hub-cache",
        str(hub_cache),
    ]
    if revision:
        cmd += ["--revision", revision]
    if max_batch_tokens is not None:
        cmd += ["--max-batch-tokens", str(max_batch_tokens)]
    return cmd


def _launch_worker(
    router_binary: pathlib.Path,
    *,
    model: str,
    revision: str | None,
    host: str,
    port: int,
    gpu_id: int,
    hub_cache: pathlib.Path,
    logs_root: pathlib.Path,
    hf_token: str | None,
    max_batch_tokens: int | None,
) -> LaunchedProcess:
    hub_cache.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"tei_gpu{gpu_id}_port{port}.log"
    uds_path = pathlib.Path(tempfile.gettempdir()) / f"pubmed-embeddings-tei-{port}.sock"
    uds_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    cmd = _build_router_command(
        router_binary,
        model=model,
        revision=revision,
        host=host,
        port=port,
        hub_cache=hub_cache,
        uds_path=uds_path,
        max_batch_tokens=max_batch_tokens,
    )
    prefix = f"tei gpu={gpu_id} port={port}"
    print(f"Starting {prefix}: {' '.join(cmd)}", file=sys.stderr)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_thread = threading.Thread(
        target=_stream_output,
        args=(prefix, process.stdout, log_path),
        daemon=True,
    )
    output_thread.start()
    return LaunchedProcess(
        gpu_id=gpu_id,
        port=port,
        base_url=f"http://{host}:{port}",
        log_path=log_path,
        process=process,
        output_thread=output_thread,
    )


def _terminate_processes(instances: list[LaunchedProcess]) -> None:
    for instance in instances:
        if instance.process.poll() is None:
            instance.process.terminate()
    deadline = time.monotonic() + 10.0
    for instance in instances:
        if instance.process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.monotonic())
        try:
            instance.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            instance.process.kill()
    for instance in instances:
        instance.output_thread.join(timeout=1.0)


def _launch_cluster(
    router_binary: pathlib.Path,
    *,
    model: str,
    revision: str | None,
    host: str,
    ports: list[int],
    gpu_ids: list[int],
    hub_cache: pathlib.Path,
    logs_root: pathlib.Path,
    hf_token: str | None,
    max_batch_tokens: int | None,
    startup_timeout: int,
) -> list[LaunchedProcess]:
    instances: list[LaunchedProcess] = []
    try:
        for gpu_id, port in zip(gpu_ids, ports):
            instance = _launch_worker(
                router_binary,
                model=model,
                revision=revision,
                host=host,
                port=port,
                gpu_id=gpu_id,
                hub_cache=hub_cache,
                logs_root=logs_root,
                hf_token=hf_token,
                max_batch_tokens=max_batch_tokens,
            )
            instances.append(instance)
            _wait_until_ready(instance, timeout_s=startup_timeout)
    except Exception:
        _terminate_processes(instances)
        raise
    return instances


def _emit_embed_instructions(*, host: str, ports: list[int], model: str) -> None:
    urls = ",".join(_router_base_urls(host, ports))
    print("", file=sys.stderr)
    print("TEI cluster is ready.", file=sys.stderr)
    print(f"  TEI_BASE_URLS={urls}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Run embeddings from another shell with:", file=sys.stderr)
    print(
        "  "
        f"EMBEDDING_SOURCE=tei-http TEI_BASE_URLS={urls} "
        f"EMBEDDING_MODEL={model} uv run pubmed-embed --data-dir data --workers {len(ports)}",
        file=sys.stderr,
    )
    print("", file=sys.stderr)


def _supervise_cluster(instances: list[LaunchedProcess]) -> int:
    while not STOP_EVENT.is_set():
        for instance in instances:
            code = instance.process.poll()
            if code is not None:
                print(
                    f"TEI worker on GPU {instance.gpu_id} port {instance.port} exited with code {code}.",
                    file=sys.stderr,
                )
                return 1
        time.sleep(1.0)
    return 130


def main(argv: Iterable[str] | None = None) -> int:
    STOP_EVENT.clear()
    args = _parse_args(argv)

    try:
        ports = _parse_csv_ints(args.ports, name="--ports")
        gpu_ids = _parse_csv_ints(args.gpu_ids, name="--gpu-ids")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if len(ports) != len(gpu_ids):
        print(
            f"--ports and --gpu-ids must have the same length; got {len(ports)} port(s) and {len(gpu_ids)} gpu id(s).",
            file=sys.stderr,
        )
        return 2

    cache_root = (args.cache_dir or _default_cache_root()).expanduser().resolve()
    paths = _managed_paths(cache_root)
    for directory in (
        paths.cache_root,
        paths.install_root,
        paths.source_root,
        paths.downloads_root,
        paths.logs_root,
        paths.hub_cache,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        visible_gpus = _query_visible_gpus()
        selected_gpus = _select_gpus(visible_gpus, gpu_ids)
        feature = _cuda_feature_for_gpus(selected_gpus)
        for gpu in selected_gpus:
            print(
                f"GPU {gpu.index}: {gpu.name} (compute capability {gpu.compute_cap})",
                file=sys.stderr,
            )
        print(f"Selected TEI CUDA feature: {feature}", file=sys.stderr)
        router_binary = _resolve_router_binary(
            paths,
            version=args.tei_version,
            feature=feature,
            require_max_batch_tokens=args.max_batch_tokens is not None,
            require_revision=args.revision is not None,
        )
    except Exception as exc:
        print(f"Failed before launch: {exc}", file=sys.stderr)
        return 1

    old_int = signal.signal(signal.SIGINT, _handle_stop_signal)
    old_term = None
    if hasattr(signal, "SIGTERM"):
        old_term = signal.signal(signal.SIGTERM, _handle_stop_signal)

    instances: list[LaunchedProcess] = []
    try:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None
        instances = _launch_cluster(
            router_binary,
            model=args.model,
            revision=args.revision,
            host=args.host,
            ports=ports,
            gpu_ids=gpu_ids,
            hub_cache=paths.hub_cache,
            logs_root=paths.logs_root,
            hf_token=hf_token,
            max_batch_tokens=args.max_batch_tokens,
            startup_timeout=args.startup_timeout,
        )

        _emit_embed_instructions(host=args.host, ports=ports, model=args.model)
        rc = _supervise_cluster(instances)
        return rc
    except KeyboardInterrupt:
        STOP_EVENT.set()
        return 130
    except Exception as exc:
        print(f"TEI cluster failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _terminate_processes(instances)
        signal.signal(signal.SIGINT, old_int)
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    raise SystemExit(main())
