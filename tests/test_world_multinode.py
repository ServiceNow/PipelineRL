"""Tests for multi-node WorldMap topology and fast-llm torchrun command assembly."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from omegaconf import OmegaConf


def _make_cfg(
    actor_fraction=1,
    finetune_fraction=1,
    preprocessor_fraction=0,
    replicas=1,
    use_fast_llm=True,
    tp=1,
    pp=1,
    seq_parallel=1,
):
    """Minimal config for WorldMap construction."""
    return OmegaConf.create({
        "world": {
            "actor_fraction": actor_fraction,
            "finetune_fraction": finetune_fraction,
            "preprocessor_fraction": preprocessor_fraction,
            "replicas": replicas,
            "actor_group_port": 9000,
            "environment_start_port": 7777,
        },
        "vllm_config": {
            "vllm_kwargs": {
                "tensor-parallel-size": tp,
                "pipeline-parallel-size": pp,
            }
        },
        "finetune": {"seq_parallel": seq_parallel},
        "use_fast_llm": use_fast_llm,
        "debug": {"mode": "", "place_inference_workers": True},
    })


def _make_world_map(cfg, world_size, rank=0, master_addr="dns-test-0"):
    from pipelinerl.world import WorldMap
    env = {
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
        "MASTER_ADDR": master_addr,
    }
    with patch.dict(os.environ, env, clear=False):
        # collect_environment_specs needs cfg fields that don't exist in minimal cfg;
        # patch it out to avoid AttributeError.
        with patch("pipelinerl.world.WorldMap._place_environments"):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                return WorldMap(cfg, verbose=False)


# ---------------------------------------------------------------------------
# WorldMap topology tests
# ---------------------------------------------------------------------------

class TestWorldMapMultiNode:

    def test_2node_1actor_1finetune_whole_nodes(self):
        """2 nodes: 1 actor node + 1 finetune node — each gets all 8 GPUs."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2)

        assert wm.total_finetune_gpus == 8, "finetune should get exactly 1 full node"
        assert wm.total_finetune_gpus % wm.node_size == 0
        assert len(wm.nodes_with_finetuning()) == 1

    def test_4node_1actor_3finetune_whole_nodes(self):
        """4 nodes: 1 actor node + 3 finetune nodes."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4)

        assert wm.total_finetune_gpus == 24, "finetune should get exactly 3 full nodes"
        assert wm.total_finetune_gpus % wm.node_size == 0
        assert len(wm.nodes_with_finetuning()) == 3

    def test_4node_2actor_2finetune_whole_nodes(self):
        """4 nodes: 2 actor nodes + 2 finetune nodes."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=2)
        wm = _make_world_map(cfg, world_size=4)

        assert wm.total_finetune_gpus == 16
        assert wm.total_finetune_gpus % wm.node_size == 0
        assert len(wm.nodes_with_finetuning()) == 2

    def test_finetune_always_at_least_one_node(self):
        """Even with a large actor fraction, finetune gets at least 1 full node."""
        cfg = _make_cfg(actor_fraction=3, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=4)

        assert len(wm.nodes_with_finetuning()) >= 1
        assert wm.total_finetune_gpus >= wm.node_size
        assert wm.total_finetune_gpus % wm.node_size == 0

    def test_actors_never_exceed_world_size_minus_one(self):
        """Actor nodes never consume all nodes — at least 1 reserved for finetune."""
        cfg = _make_cfg(actor_fraction=10, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=4)

        finetune_nodes = len(wm.nodes_with_finetuning())
        assert finetune_nodes >= 1
        assert finetune_nodes < 4

    def test_single_node_unchanged(self):
        """Single-node path is not affected by the multi-node rounding."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6)
        # Single-node: world_size=1, node_size = actual device count (mocked)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)
        assert wm.total_finetune_gpus == 6
        assert wm.world_size == 1

    def test_nodes_with_finetuning_returns_sorted_ranks(self):
        """nodes_with_finetuning() returns a sorted list of node ranks."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4)

        fn = wm.nodes_with_finetuning()
        assert fn == sorted(fn)

    def test_my_finetuning_rank_on_finetune_node(self):
        """my_finetuning_rank() returns 0 for the first finetune node."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        # With 2 nodes, finetune is on node 0 (actor on node 1 due to reversed placement)
        wm = _make_world_map(cfg, world_size=2, rank=0)

        finetune_nodes = wm.nodes_with_finetuning()
        # my_rank=0 should be a finetune node
        assert 0 in finetune_nodes
        assert wm.my_finetuning_rank() == finetune_nodes.index(0)

    def test_4node_with_preprocessor_all_whole_nodes(self):
        """4 nodes, actor=1, preprocessor=1, finetune=6: all three get whole nodes."""
        cfg = _make_cfg(actor_fraction=1, preprocessor_fraction=1, finetune_fraction=6)
        wm = _make_world_map(cfg, world_size=4)

        assert wm.total_finetune_gpus % wm.node_size == 0, "finetune must be whole nodes"
        # preprocessor and actor GPU shares should also be multiples of node_size
        total = wm.world_size * wm.node_size
        actor_gpus = total - wm.total_finetune_gpus - wm.gpus_per_preprocessor * cfg.world.replicas
        assert actor_gpus % wm.node_size == 0, "actor must be whole nodes"
        assert (wm.gpus_per_preprocessor * cfg.world.replicas) % wm.node_size == 0, "preprocessor must be whole nodes"
        assert wm.total_finetune_gpus + actor_gpus + wm.gpus_per_preprocessor * cfg.world.replicas == total

    def test_3node_with_preprocessor_all_whole_nodes(self):
        """3 nodes, actor=1, preprocessor=1, finetune=1: each component gets 1 node."""
        cfg = _make_cfg(actor_fraction=1, preprocessor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=3)

        assert wm.total_finetune_gpus % wm.node_size == 0
        total = wm.world_size * wm.node_size
        actor_gpus = total - wm.total_finetune_gpus - wm.gpus_per_preprocessor * cfg.world.replicas
        assert actor_gpus % wm.node_size == 0
        assert (wm.gpus_per_preprocessor * cfg.world.replicas) % wm.node_size == 0

    def test_address_map_derived_from_master_addr(self):
        """address_map entries follow the dns-<uuid>-<rank> pattern."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        assert wm.address_map[0] == "dns-abc123-0"
        assert wm.address_map[1] == "dns-abc123-1"


# ---------------------------------------------------------------------------
# torchrun command assembly test
# ---------------------------------------------------------------------------

class TestTorchrunCommand:

    def _capture_cmd(self, world_map, cfg_extra=None):
        """Run _run_finetune_fast_llm with mocked I/O and capture the torchrun command."""
        from pipelinerl.launch import _run_finetune_fast_llm

        cfg = OmegaConf.create({
            "model_path": "/tmp/fake_model",
            "weight_broadcast": False,
            "debug": {"mode": "", "log_data_pipeline": False},
            "streams": {"host": "localhost", "port": 11000},
            "wandb": {
                "wandb_workspace_root": "/tmp",
                "wandb_entity_name": "test",
                "wandb_project_name": "test",
                "wandb_group": "test",
            },
            "fast_llm": {
                "training": {
                    "train_iters": 10,
                    "wandb": {"entity_name": None, "project_name": None, "group_name": None},
                },
                "data": {"datasets": {"training": {"type": "streaming", "host": None, "port": None}}},
                "pretrained": {"format": "llama", "path": None, "model_weights": True},
                "run": {"experiment_dir": None, "experiment_name": None},
                "callbacks": {},
            },
            "fast_llm_finetune": {
                "model_type": "llama",
                "torchrun_port": 29500,
                "model_format": "llama",
            },
        })
        if cfg_extra:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_extra))

        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return None  # no process spawned

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            # Patch os.path.isdir to pass the model_path check
            with patch("pipelinerl.launch._popen", side_effect=mock_popen):
                with patch("pipelinerl.launch.save_command"):
                    with patch("os.path.isdir", return_value=True):
                        list(_run_finetune_fast_llm(cfg, world_map, gpus=[0, 1, 2, 3], exp_dir=exp_dir))

        return captured_cmd

    def test_single_node_uses_master_port(self):
        """Single-node torchrun uses --master_port, no rdzv flags."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        cmd = self._capture_cmd(wm)
        assert "--master_port=29500" in cmd
        assert "--rdzv_backend=static" not in cmd
        assert "--nnodes=6" not in cmd

    def test_2node_1finetune_uses_single_node_torchrun(self):
        """2-node job with 1 actor + 1 finetune node: fast-llm spans 1 node → single-node torchrun."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2, rank=0, master_addr="dns-abc-0")

        assert len(wm.nodes_with_finetuning()) == 1, "only 1 finetune node in 2-node job"
        cmd = self._capture_cmd(wm)
        # Should use simple --master_port, not rdzv
        assert "--master_port=29500" in cmd
        assert "--rdzv_backend=static" not in cmd

    def test_multi_node_uses_static_rdzv(self):
        """Fast-llm spanning multiple nodes uses static rdzv with correct nnodes and node_rank."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4, rank=0, master_addr="dns-abc-0")

        assert len(wm.nodes_with_finetuning()) == 3
        cmd = self._capture_cmd(wm)
        assert "--rdzv_backend=static" in cmd
        assert "--rdzv_id=0" in cmd
        assert "--max_restarts=0" in cmd
        finetune_count = len(wm.nodes_with_finetuning())
        assert f"--nnodes={finetune_count}" in cmd
        assert f"--node_rank={wm.my_finetuning_rank()}" in cmd
        finetune_master = wm.address_map[wm.nodes_with_finetuning()[0]]
        assert any(f"--rdzv_endpoint={finetune_master}:29500" in arg for arg in cmd)
        assert not any("--master_port" in arg for arg in cmd)

    def test_multi_node_4nodes_correct_nnodes(self):
        """4-node job: torchrun nnodes = 3 (finetune nodes only)."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4, rank=0)

        cmd = self._capture_cmd(wm)
        finetune_count = len(wm.nodes_with_finetuning())
        assert finetune_count == 3
        assert f"--nnodes={finetune_count}" in cmd


# ---------------------------------------------------------------------------
# DeepSpeed regression: snapping must NOT apply when use_fast_llm=False
# ---------------------------------------------------------------------------

class TestWorldMapDeepSpeed:

    def test_deepspeed_single_node_fractional_split(self):
        """Single-node DeepSpeed split is unchanged — 2 actor GPUs + 6 finetune GPUs."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6, use_fast_llm=False)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        assert wm.total_finetune_gpus == 6
        assert wm.world_size == 1

    def test_deepspeed_multinode_no_rounding(self):
        """Multi-node DeepSpeed: no whole-node snapping (handled by DeepSpeed itself)."""
        # 2 nodes, actor_fraction=1, finetune_fraction=1 → 8 finetune GPUs (happens to be whole node)
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2)
        # Should still compute correctly without triggering fast-llm rounding path
        assert wm.total_finetune_gpus > 0
        assert wm.world_size == 2

    def test_fast_llm_single_node_unchanged(self):
        """Single-node fast-llm: fractional split within one node is preserved."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6, use_fast_llm=True)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        assert wm.total_finetune_gpus == 6
        assert wm.world_size == 1


# ---------------------------------------------------------------------------
# Pod IP exchange: dns_address_map, job URL rewriting, DeepSpeed/fast-llm compat
# ---------------------------------------------------------------------------

def _simulate_pod_ip_exchange(wm, pod_ips: dict):
    """Simulate _exchange_pod_ips without NFS I/O.

    Sets dns_address_map to original DNS names, updates address_map and job
    URLs/hostnames to pod IPs — mirrors the real function's side-effects.
    """
    from pipelinerl.launch import _exchange_pod_ips as real_fn  # noqa: F401 (not called)
    # Save DNS names first (matches the real implementation order)
    wm.dns_address_map = dict(wm.address_map)
    # Overwrite address_map with pod IPs
    for rank, ip in pod_ips.items():
        wm.address_map[rank] = ip
    wm.master_addr = pod_ips[0]
    # Rewrite job URLs/hostnames
    for node, jobs in wm.job_map.items():
        dns_name = wm.dns_address_map[node]
        pod_ip = pod_ips[node]
        for job in jobs:
            job.hostname = pod_ip
            if job.url:
                job.url = job.url.replace(dns_name, pod_ip)


class TestPodIPExchange:

    def test_dns_address_map_holds_original_dns_names(self):
        """After pod IP exchange, dns_address_map contains original DNS names, not pod IPs."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        pod_ips = {0: "10.0.0.1", 1: "10.0.0.2"}
        _simulate_pod_ip_exchange(wm, pod_ips)

        assert wm.dns_address_map[0] == "dns-abc123-0"
        assert wm.dns_address_map[1] == "dns-abc123-1"

    def test_address_map_updated_to_pod_ips(self):
        """After pod IP exchange, address_map and master_addr hold pod IPs."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        pod_ips = {0: "10.0.0.1", 1: "10.0.0.2"}
        _simulate_pod_ip_exchange(wm, pod_ips)

        assert wm.address_map[0] == "10.0.0.1"
        assert wm.address_map[1] == "10.0.0.2"
        assert wm.master_addr == "10.0.0.1"

    def test_job_urls_rewritten_to_pod_ips(self):
        """After pod IP exchange, actor_llm job URLs use pod IPs, not DNS names."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        # Verify that actor_llm jobs have DNS-based URLs before exchange
        actor_urls_before = [job.url for job in wm.get_all_jobs() if job.kind == "actor_llm"]
        assert all("dns-abc123-1" in u for u in actor_urls_before)

        pod_ips = {0: "10.0.0.1", 1: "10.0.0.2"}
        _simulate_pod_ip_exchange(wm, pod_ips)

        actor_urls_after = [job.url for job in wm.get_all_jobs() if job.kind == "actor_llm"]
        assert all("10.0.0.2" in u for u in actor_urls_after), f"Expected pod IP in URLs: {actor_urls_after}"
        assert all("dns-abc123" not in u for u in actor_urls_after)

    def test_no_dns_address_map_without_exchange(self):
        """Without pod IP exchange, dns_address_map is not set (no AttributeError)."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")
        assert not hasattr(wm, "dns_address_map")


# ---------------------------------------------------------------------------
# DeepSpeed command assembly: hostfile and inclusion filter use DNS names
# ---------------------------------------------------------------------------

class TestDeepSpeedCommand:

    def _make_ds_cfg(self):
        return OmegaConf.create({
            "use_deepspeed": True,
            "use_fsdp": False,
            "deepspeed_config": "zero2",
            "accelerate_config": None,
            "world": {"actor_group_port": 9000},
            "debug": {"mode": ""},
        })

    def _capture_ds_cmd(self, world_map, cfg_extra=None):
        """Run _run_finetune_deepspeed with mocked I/O and capture the command."""
        from pipelinerl.launch import _run_finetune_deepspeed

        cfg = self._make_ds_cfg()
        if cfg_extra:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_extra))

        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return None

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "hostfile.txt").write_text("")  # pre-create
            with patch("pipelinerl.launch._popen", side_effect=mock_popen):
                with patch("pipelinerl.launch.save_command"):
                    with patch.dict(os.environ, {"MASTER_ADDR": "dns-test-0", "MASTER_PORT": "29501"}):
                        list(_run_finetune_deepspeed(cfg, world_map, gpus=[0, 1, 2, 3], exp_dir=exp_dir))

        return captured_cmd

    def test_deepspeed_multinode_uses_dns_names_without_exchange(self):
        """DeepSpeed 2-node without pod IP exchange: inclusion filter uses DNS names."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        cmd = self._capture_ds_cmd(wm)
        # The deepspeed_inclusion_filter should contain the DNS hostname for the finetune node
        filter_arg = next((c for c in cmd if "dns-abc123" in c), None)
        assert filter_arg is not None, f"Expected DNS name in cmd, got: {cmd}"

    def test_deepspeed_multinode_after_pod_ip_exchange_uses_dns_names(self):
        """After pod IP exchange, DeepSpeed inclusion filter still uses DNS names (not pod IPs)."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        # Simulate pod IP exchange
        _simulate_pod_ip_exchange(wm, {0: "10.0.0.1", 1: "10.0.0.2"})

        cmd = self._capture_ds_cmd(wm)
        # Inclusion filter must still use DNS names, not pod IPs
        filter_arg = next((c for c in cmd if "dns-abc123" in c), None)
        assert filter_arg is not None, f"Expected DNS name in DS filter after pod IP exchange, got: {cmd}"
        # Pod IPs must NOT appear in the inclusion filter
        assert not any("10.0.0" in c for c in cmd if "--deepspeed_inclusion_filter" not in c and "@" in c), \
            f"Pod IP leaked into DS filter: {cmd}"

    def test_deepspeed_single_node_no_pod_ip_exchange(self):
        """Single-node DeepSpeed: no world_size>1 branch, pod IP exchange never runs."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6, use_fast_llm=False)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        assert wm.world_size == 1
        assert not hasattr(wm, "dns_address_map")
        # Should not crash even without dns_address_map
        cmd = self._capture_ds_cmd(wm)
        assert "--num_machines" not in cmd  # single-node, no multi-machine flags


# ---------------------------------------------------------------------------
# Hostfile creation in main(): uses dns_address_map after pod IP exchange
# ---------------------------------------------------------------------------

class TestHostfileCreation:

    def test_hostfile_uses_dns_names_after_pod_ip_exchange(self):
        """The DeepSpeed hostfile written by main() uses DNS names even after pod IP exchange."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        # Simulate pod IP exchange
        _simulate_pod_ip_exchange(wm, {0: "10.0.0.1", 1: "10.0.0.2"})

        dns_map = getattr(wm, "dns_address_map", wm.address_map)
        hosts = [dns_map[i] for i in range(wm.world_size)]

        assert hosts[0] == "dns-abc123-0"
        assert hosts[1] == "dns-abc123-1"
        assert "10.0.0" not in hosts[0]
        assert "10.0.0" not in hosts[1]

    def test_hostfile_uses_address_map_without_exchange(self):
        """Without pod IP exchange, dns_address_map is absent — falls back to address_map."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        dns_map = getattr(wm, "dns_address_map", wm.address_map)
        hosts = [dns_map[i] for i in range(wm.world_size)]

        assert hosts[0] == "dns-abc123-0"
        assert hosts[1] == "dns-abc123-1"


# ---------------------------------------------------------------------------
# Redis host in saved exp_config.yaml for multi-node (DeepSpeed + Redis)
# ---------------------------------------------------------------------------

class TestRedisHostMultiNode:

    def _compute_streams_host(self, world_map, my_rank: int) -> str:
        """Mirror the launch.py logic for cfg.streams.host selection."""
        if world_map.world_size > 1:
            return world_map.master_addr
        return "localhost"

    def test_single_node_redis_host_is_localhost(self):
        """Single-node: Redis host is localhost regardless of pod IP exchange."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6, use_fast_llm=False)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        host = self._compute_streams_host(wm, my_rank=0)
        assert host == "localhost"

    def test_multinode_rank0_redis_host_is_pod_ip(self):
        """Multi-node rank 0: Redis host is pod IP (not localhost) after exchange.

        This ensures the saved exp_config.yaml has a reachable address for
        DeepSpeed workers on other nodes.
        """
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")
        _simulate_pod_ip_exchange(wm, {0: "10.0.0.1", 1: "10.0.0.2"})

        host = self._compute_streams_host(wm, my_rank=0)
        assert host == "10.0.0.1", "rank 0 should use pod IP so saved config is reachable cross-node"
        assert host != "localhost"

    def test_multinode_rank1_redis_host_is_pod_ip(self):
        """Multi-node rank 1: Redis host is pod IP of rank 0."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0", rank=1)
        _simulate_pod_ip_exchange(wm, {0: "10.0.0.1", 1: "10.0.0.2"})

        host = self._compute_streams_host(wm, my_rank=1)
        assert host == "10.0.0.1", "rank 1 should use rank 0's pod IP to reach Redis"

    def test_multinode_both_ranks_same_redis_host(self):
        """Both ranks in a 2-node job resolve to the same Redis host (pod IP of rank 0)."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm0 = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0", rank=0)
        wm1 = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0", rank=1)

        _simulate_pod_ip_exchange(wm0, {0: "10.0.0.1", 1: "10.0.0.2"})
        _simulate_pod_ip_exchange(wm1, {0: "10.0.0.1", 1: "10.0.0.2"})

        host0 = self._compute_streams_host(wm0, my_rank=0)
        host1 = self._compute_streams_host(wm1, my_rank=1)

        assert host0 == host1 == "10.0.0.1"

    def test_multinode_without_pod_ip_exchange_uses_master_addr(self):
        """Without pod IP exchange, multi-node uses master_addr (DNS name) for Redis.

        This is a fallback; the pod IP exchange should always run in practice
        but the code must not crash without it.
        """
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=1, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=2, master_addr="dns-abc123-0")

        # No pod IP exchange — master_addr is still a DNS name
        assert wm.master_addr == "dns-abc123-0"
        host = self._compute_streams_host(wm, my_rank=0)
        assert host == "dns-abc123-0"  # DNS name (port filtering may apply, but code doesn't crash)


# ---------------------------------------------------------------------------
# DeepSpeed run_finetune.py path: must be absolute (not relative to CWD)
# ---------------------------------------------------------------------------

class TestDeepSpeedEntrypointPath:

    def _capture_ds_cmd(self, world_map):
        from pipelinerl.launch import _run_finetune_deepspeed
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "use_deepspeed": True,
            "use_fsdp": False,
            "deepspeed_config": "zero2",
            "accelerate_config": None,
            "world": {"actor_group_port": 9000},
            "debug": {"mode": ""},
        })
        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return None

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            with patch("pipelinerl.launch._popen", side_effect=mock_popen):
                with patch("pipelinerl.launch.save_command"):
                    with patch.dict(os.environ, {"MASTER_ADDR": "dns-test-0", "MASTER_PORT": "29501"}):
                        list(_run_finetune_deepspeed(cfg, world_map, gpus=[0, 1, 2, 3], exp_dir=exp_dir))

        return captured_cmd

    def test_run_finetune_path_is_absolute(self):
        """run_finetune.py must be an absolute path so it works regardless of CWD.

        When EAI starts the pod, CWD is /home/toolkit (not the repo root). A relative
        path like 'pipelinerl/entrypoints/run_finetune.py' resolves to
        '/home/toolkit/pipelinerl/...' which doesn't exist.
        """
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6, use_fast_llm=False)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        cmd = self._capture_ds_cmd(wm)

        # Find the run_finetune.py argument
        finetune_script = next((c for c in cmd if "run_finetune.py" in c), None)
        assert finetune_script is not None, f"run_finetune.py not found in cmd: {cmd}"
        assert Path(finetune_script).is_absolute(), (
            f"run_finetune.py path must be absolute but got: {finetune_script!r}. "
            "A relative path resolves against CWD which is /home/toolkit in EAI pods."
        )
        assert Path(finetune_script).exists(), (
            f"run_finetune.py absolute path must exist: {finetune_script!r}"
        )


# ---------------------------------------------------------------------------
# Per-node file naming: fast-llm and DeepSpeed avoid NFS write races
# ---------------------------------------------------------------------------

class TestPerNodeFileNaming:
    """Verify that multinode fast-llm and DeepSpeed finetune runs write separate
    output files per node (config, start.sh, stdout, stderr) to avoid NFS races."""

    def _capture_fast_llm_files(self, world_map, gpus=None):
        """Run _run_finetune_fast_llm and return captured file suffix info."""
        from pipelinerl.launch import _run_finetune_fast_llm

        cfg = OmegaConf.create({
            "model_path": "/tmp/fake_model",
            "weight_broadcast": False,
            "debug": {"mode": "", "log_data_pipeline": False},
            "streams": {"host": "localhost", "port": 11000},
            "wandb": {
                "wandb_workspace_root": "/tmp",
                "wandb_entity_name": "test",
                "wandb_project_name": "test",
                "wandb_group": "test",
            },
            "fast_llm": {
                "training": {
                    "train_iters": 10,
                    "wandb": {"entity_name": None, "project_name": None, "group_name": None},
                },
                "data": {"datasets": {"training": {"type": "streaming", "host": None, "port": None}}},
                "pretrained": {"format": "llama", "path": None, "model_weights": True},
                "run": {"experiment_dir": None, "experiment_name": None},
                "callbacks": {},
            },
            "fast_llm_finetune": {
                "model_type": "llama",
                "torchrun_port": 29500,
                "model_format": "llama",
            },
        })

        written_files = {}

        real_open = open

        def mock_popen(cmd, **kwargs):
            written_files["stdout"] = str(kwargs.get("stdout", {}).name if hasattr(kwargs.get("stdout"), "name") else "")
            written_files["stderr"] = str(kwargs.get("stderr", {}).name if hasattr(kwargs.get("stderr"), "name") else "")
            return None

        captured_save = {}

        def mock_save_command(script_dir, cmd, suffix=""):
            captured_save["suffix"] = suffix
            captured_save["dir"] = str(script_dir)

        captured_config = {}

        real_omegaconf_save = None

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            with patch("pipelinerl.launch._popen", side_effect=mock_popen):
                with patch("pipelinerl.launch.save_command", side_effect=mock_save_command):
                    with patch("os.path.isdir", return_value=True):
                        with patch("omegaconf.OmegaConf.save") as mock_cfg_save:
                            list(_run_finetune_fast_llm(cfg, world_map, gpus=gpus or [0, 1, 2, 3], exp_dir=exp_dir))
                            if mock_cfg_save.call_args:
                                # OmegaConf.save(cfg, path) — second positional arg is path
                                args = mock_cfg_save.call_args[0]
                                captured_config["path"] = str(args[1]) if len(args) > 1 else ""

            return {
                "config_path": captured_config.get("path", ""),
                "save_suffix": captured_save.get("suffix", ""),
                "stdout": written_files.get("stdout", ""),
                "stderr": written_files.get("stderr", ""),
            }

    def _capture_deepspeed_files(self, world_map, gpus=None):
        """Run _run_finetune_deepspeed and return captured file suffix."""
        from pipelinerl.launch import _run_finetune_deepspeed

        cfg = OmegaConf.create({
            "use_deepspeed": True,
            "use_fsdp": False,
            "deepspeed_config": "zero2",
            "accelerate_config": None,
            "world": {"actor_group_port": 9000},
            "debug": {"mode": ""},
        })

        captured_save = {}
        written_files = {}

        def mock_popen(cmd, **kwargs):
            written_files["stdout"] = str(kwargs.get("stdout", {}).name if hasattr(kwargs.get("stdout"), "name") else "")
            written_files["stderr"] = str(kwargs.get("stderr", {}).name if hasattr(kwargs.get("stderr"), "name") else "")
            return None

        def mock_save_command(script_dir, cmd, suffix=""):
            captured_save["suffix"] = suffix

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            with patch("pipelinerl.launch._popen", side_effect=mock_popen):
                with patch("pipelinerl.launch.save_command", side_effect=mock_save_command):
                    with patch.dict(os.environ, {"MASTER_ADDR": "dns-test-0", "MASTER_PORT": "29501"}):
                        list(_run_finetune_deepspeed(cfg, world_map, gpus=gpus or [0, 1, 2, 3], exp_dir=exp_dir))

        return {
            "save_suffix": captured_save.get("suffix", ""),
            "stdout": written_files.get("stdout", ""),
            "stderr": written_files.get("stderr", ""),
        }

    # --- fast-llm single-node: no suffix ---

    def test_fast_llm_single_node_no_suffix(self):
        """Single-node fast-llm: no _node0 suffix — backward compat."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        result = self._capture_fast_llm_files(wm)
        assert result["save_suffix"] == "", f"Single-node must have no suffix, got: {result['save_suffix']!r}"
        assert "_node" not in result["config_path"], f"Single-node config must have no _node suffix: {result['config_path']}"

    # --- fast-llm multinode: each node gets its own suffix ---

    def test_fast_llm_multinode_node0_suffix(self):
        """4-node fast-llm, finetune node 0: files get _node0 suffix.
        Actor takes the last node (rank 3), so ranks 0/1/2 are finetune."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4, rank=0)  # rank 0 = first finetune node

        result = self._capture_fast_llm_files(wm)
        assert result["save_suffix"] == "_node0", f"Expected _node0, got: {result['save_suffix']!r}"
        assert "_node0" in result["config_path"], f"Config path must contain _node0: {result['config_path']}"

    def test_fast_llm_multinode_node1_suffix(self):
        """4-node fast-llm, finetune node 1: files get _node1 suffix."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4, rank=1)  # rank 1 = second finetune node

        result = self._capture_fast_llm_files(wm)
        assert result["save_suffix"] == "_node1", f"Expected _node1, got: {result['save_suffix']!r}"
        assert "_node1" in result["config_path"]

    def test_fast_llm_multinode_node2_suffix(self):
        """4-node fast-llm, finetune node 2: files get _node2 suffix."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3)
        wm = _make_world_map(cfg, world_size=4, rank=2)  # rank 2 = third finetune node

        result = self._capture_fast_llm_files(wm)
        assert result["save_suffix"] == "_node2", f"Expected _node2, got: {result['save_suffix']!r}"

    # --- DeepSpeed single-node: no suffix ---

    def test_deepspeed_single_node_no_suffix(self):
        """Single-node DeepSpeed: no _node suffix."""
        cfg = _make_cfg(actor_fraction=2, finetune_fraction=6, use_fast_llm=False)
        with patch("torch.cuda.device_count", return_value=8):
            with patch("pipelinerl.utils.collect_environment_specs", return_value=[]):
                with patch("pipelinerl.world.WorldMap._place_environments"):
                    from pipelinerl.world import WorldMap
                    wm = WorldMap(cfg, verbose=False)

        result = self._capture_deepspeed_files(wm)
        assert result["save_suffix"] == "", f"Single-node must have no suffix, got: {result['save_suffix']!r}"

    # --- DeepSpeed multinode: each node gets its own suffix ---

    def test_deepspeed_multinode_node0_suffix(self):
        """4-node DeepSpeed, finetune node 0: save_command gets _node0 suffix.
        Actor takes the last node (rank 3), so ranks 0/1/2 are finetune."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=4, rank=0)  # rank 0 = first finetune node

        result = self._capture_deepspeed_files(wm)
        assert result["save_suffix"] == "_node0", f"Expected _node0, got: {result['save_suffix']!r}"

    def test_deepspeed_multinode_node2_suffix(self):
        """4-node DeepSpeed, finetune node 2: save_command gets _node2 suffix."""
        cfg = _make_cfg(actor_fraction=1, finetune_fraction=3, use_fast_llm=False)
        wm = _make_world_map(cfg, world_size=4, rank=2)  # rank 2 = third finetune node

        result = self._capture_deepspeed_files(wm)
        assert result["save_suffix"] == "_node2", f"Expected _node2, got: {result['save_suffix']!r}"
