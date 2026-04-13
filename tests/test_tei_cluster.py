from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from pubmed_embeddings import tei_cluster


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300


class _FakeClient:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self._responses = responses

    def get(self, url: str) -> _FakeResponse:
        return self._responses[url]


class TeiClusterTests(unittest.TestCase):
    def test_parse_csv_ints(self) -> None:
        self.assertEqual(tei_cluster._parse_csv_ints("11450, 11451,11452", name="ports"), [11450, 11451, 11452])

    def test_parse_csv_ints_rejects_empty_items(self) -> None:
        with self.assertRaises(ValueError):
            tei_cluster._parse_csv_ints("11450,,11452", name="ports")

    def test_select_gpus_preserves_requested_order(self) -> None:
        gpus = [
            tei_cluster.GpuInfo(index=0, name="GPU-0", compute_cap=8.0),
            tei_cluster.GpuInfo(index=1, name="GPU-1", compute_cap=8.0),
            tei_cluster.GpuInfo(index=2, name="GPU-2", compute_cap=8.0),
        ]
        selected = tei_cluster._select_gpus(gpus, [2, 0])
        self.assertEqual([gpu.index for gpu in selected], [2, 0])

    def test_cuda_feature_for_ampere_or_newer(self) -> None:
        feature = tei_cluster._cuda_feature_for_gpus(
            [
                tei_cluster.GpuInfo(index=0, name="A10", compute_cap=8.6),
                tei_cluster.GpuInfo(index=1, name="L40", compute_cap=8.9),
            ]
        )
        self.assertEqual(feature, "candle-cuda")

    def test_cuda_feature_for_turing(self) -> None:
        feature = tei_cluster._cuda_feature_for_gpus(
            [tei_cluster.GpuInfo(index=0, name="T4", compute_cap=7.5)]
        )
        self.assertEqual(feature, "candle-cuda-turing")

    def test_cuda_feature_rejects_mixed_families(self) -> None:
        with self.assertRaises(RuntimeError):
            tei_cluster._cuda_feature_for_gpus(
                [
                    tei_cluster.GpuInfo(index=0, name="T4", compute_cap=7.5),
                    tei_cluster.GpuInfo(index=1, name="A10", compute_cap=8.6),
                ]
            )

    def test_build_router_command_contains_required_flags(self) -> None:
        command = tei_cluster._build_router_command(
            tei_cluster.pathlib.Path("/tmp/text-embeddings-router"),
            model="BAAI/bge-large-en-v1.5",
            revision="main",
            host="127.0.0.1",
            port=11450,
            hub_cache=tei_cluster.pathlib.Path("/tmp/hf-cache"),
            uds_path=tei_cluster.pathlib.Path("/tmp/tei.sock"),
            max_batch_tokens=32768,
        )
        self.assertEqual(command[0], "/tmp/text-embeddings-router")
        self.assertIn("--model-id", command)
        self.assertIn("BAAI/bge-large-en-v1.5", command)
        self.assertIn("--hostname", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--port", command)
        self.assertIn("11450", command)
        self.assertIn("--uds-path", command)
        self.assertIn("/tmp/tei.sock", command)
        self.assertIn("--max-batch-tokens", command)
        self.assertIn("32768", command)

    def test_health_check_uses_health_endpoint_first(self) -> None:
        base_url = "http://127.0.0.1:11450"
        client = _FakeClient(
            {
                f"{base_url}/health": _FakeResponse(200),
                f"{base_url}/docs": _FakeResponse(200),
            }
        )
        ok, detail = tei_cluster._health_check(client, base_url)
        self.assertTrue(ok)
        self.assertEqual(detail, "/health")

    def test_health_check_falls_back_to_docs(self) -> None:
        base_url = "http://127.0.0.1:11450"
        client = _FakeClient(
            {
                f"{base_url}/health": _FakeResponse(404),
                f"{base_url}/docs": _FakeResponse(200),
            }
        )
        ok, detail = tei_cluster._health_check(client, base_url)
        self.assertTrue(ok)
        self.assertEqual(detail, "/docs")

    def test_probe_router_binary_accepts_matching_binary(self) -> None:
        help_proc = subprocess.CompletedProcess(
            args=["router", "--help"],
            returncode=0,
            stdout=" ".join(
                [
                    "--model-id",
                    "--hostname",
                    "--port",
                    "--uds-path",
                    "--huggingface-hub-cache",
                    "--revision",
                    "--max-batch-tokens",
                ]
            ),
            stderr="",
        )
        version_proc = subprocess.CompletedProcess(
            args=["router", "--version"],
            returncode=0,
            stdout="text-embeddings-router 1.8.3",
            stderr="",
        )
        with mock.patch("subprocess.run", side_effect=[help_proc, version_proc]):
            detected = tei_cluster._probe_router_binary(
                pathlib.Path("/usr/bin/text-embeddings-router"),
                expected_version="v1.8.3",
                require_max_batch_tokens=True,
                require_revision=True,
            )
        self.assertEqual(detected, "1.8.3")

    def test_probe_router_binary_rejects_missing_flags(self) -> None:
        help_proc = subprocess.CompletedProcess(
            args=["router", "--help"],
            returncode=0,
            stdout="--model-id --hostname --port",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=help_proc):
            with self.assertRaises(RuntimeError):
                tei_cluster._probe_router_binary(
                    pathlib.Path("/usr/bin/text-embeddings-router"),
                    expected_version="v1.8.3",
                    require_max_batch_tokens=False,
                    require_revision=False,
                )

    def test_resolve_toolchain_installs_when_cargo_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = tei_cluster._managed_paths(pathlib.Path(tmp))
            with mock.patch("shutil.which", return_value=None):
                fake_toolchain = tei_cluster.Toolchain(
                    cargo=pathlib.Path("/tmp/cargo"),
                    env_overrides={"PATH": "/tmp"},
                )
                with mock.patch(
                    "pubmed_embeddings.tei_cluster._install_managed_rustup",
                    return_value=fake_toolchain,
                ) as install:
                    toolchain = tei_cluster._resolve_toolchain(paths)
        self.assertEqual(toolchain, fake_toolchain)
        install.assert_called_once()

    def test_verify_sha256_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "file.bin"
            path.write_bytes(b"content")
            with self.assertRaises(RuntimeError):
                tei_cluster._verify_sha256(path, "deadbeef")

    def test_candidate_cuda_bin_dirs_includes_versioned_usr_local_paths(self) -> None:
        env = {"PATH": ""}
        fake_path = pathlib.Path("/usr/local/cuda-12.6/bin")
        original_glob = pathlib.Path.glob

        def fake_glob(path_obj: pathlib.Path, pattern: str):  # type: ignore[override]
            if path_obj == pathlib.Path("/usr/local") and pattern == "cuda-*/bin":
                return [fake_path]
            return list(original_glob(path_obj, pattern))

        with mock.patch("pathlib.Path.glob", autospec=True, side_effect=fake_glob):
            with mock.patch("pathlib.Path.exists", autospec=True) as exists:
                exists.side_effect = lambda p: p == fake_path
                candidates = tei_cluster._candidate_cuda_bin_dirs(env)
        self.assertIn(fake_path, candidates)

    def test_resolve_router_binary_falls_back_when_path_binary_is_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = tei_cluster._managed_paths(pathlib.Path(tmp))
            managed = pathlib.Path(tmp) / "managed" / "text-embeddings-router"
            with mock.patch("shutil.which", return_value="/usr/bin/text-embeddings-router"):
                with mock.patch(
                    "pubmed_embeddings.tei_cluster._probe_router_binary",
                    side_effect=RuntimeError("bad help output"),
                ):
                    with mock.patch(
                        "pubmed_embeddings.tei_cluster._ensure_managed_router_binary",
                        return_value=managed,
                    ) as ensure_managed:
                        resolved = tei_cluster._resolve_router_binary(
                            paths,
                            version="v1.8.3",
                            feature="candle-cuda",
                            require_max_batch_tokens=False,
                            require_revision=False,
                        )
        self.assertEqual(resolved, managed)
        ensure_managed.assert_called_once()

    def test_launch_cluster_terminates_started_workers_on_readiness_failure(self) -> None:
        fake_instances = [
            tei_cluster.LaunchedProcess(
                gpu_id=0,
                port=11450,
                base_url="http://127.0.0.1:11450",
                log_path=pathlib.Path("/tmp/tei0.log"),
                process=mock.Mock(),
                output_thread=mock.Mock(),
            ),
            tei_cluster.LaunchedProcess(
                gpu_id=1,
                port=11451,
                base_url="http://127.0.0.1:11451",
                log_path=pathlib.Path("/tmp/tei1.log"),
                process=mock.Mock(),
                output_thread=mock.Mock(),
            ),
        ]
        with mock.patch(
            "pubmed_embeddings.tei_cluster._launch_worker",
            side_effect=fake_instances,
        ):
            with mock.patch(
                "pubmed_embeddings.tei_cluster._wait_until_ready",
                side_effect=[None, RuntimeError("worker failed")],
            ):
                with mock.patch(
                    "pubmed_embeddings.tei_cluster._terminate_processes"
                ) as terminate:
                    with self.assertRaises(RuntimeError):
                        tei_cluster._launch_cluster(
                            pathlib.Path("/usr/bin/text-embeddings-router"),
                            model="BAAI/bge-large-en-v1.5",
                            revision=None,
                            host="127.0.0.1",
                            ports=[11450, 11451],
                            gpu_ids=[0, 1],
                            hub_cache=pathlib.Path("/tmp/hf"),
                            logs_root=pathlib.Path("/tmp/logs"),
                            hf_token=None,
                            max_batch_tokens=None,
                            startup_timeout=60,
                        )
        terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
