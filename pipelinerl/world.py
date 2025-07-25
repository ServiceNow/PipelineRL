import logging
import os
from typing import Literal
from pydantic import BaseModel
from omegaconf import DictConfig
import torch

logger = logging.getLogger(__name__)


class Job(BaseModel):
    """Represent the decision to launch a replica of a particular worker (e.g. actor) at a particular rank"""
    # The job kind 
    kind: str
    # The global index of this job among all jobs
    idx: int 
    # The index of this job among jobs of the same kind
    replica_idx: int
    # The index of this job among similar jobs on the same node
    local_idx: int = 0
    # Where this job should run
    node_rank: int
    hostname: str 
    port: int | None = None
    # Which GPUs the job will use
    gpus: list[int] = []
    # The URL of the job
    url: str = ""


class WorldMap:
    def __init__(self, cfg: DictConfig, verbose: bool = False):
        self._log_info = logger.info if verbose else lambda x: None

        self.cfg = cfg
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.my_rank = int(os.environ.get("RANK", 0))
        self.address_map = {}
        if self.world_size > 1:
            self.master_addr = os.environ["MASTER_ADDR"]
            # e.g.: dns-f6c9712f-4d9b-4c8d-a648-f8d94cf12113-0
            for rank in range(self.world_size):
                basename = self.master_addr[: self.master_addr.rfind("-")]
                self.address_map[rank] = f"{basename}-{rank}"
        else:
            self.master_addr = "localhost"
            self.address_map[0] = "localhost"

        self._log_info(f"--- INITIALIZE WORLD MAP (this is rank {self.my_rank}) ---")

        llm_kwargs = self.cfg.vllm_config.vllm_kwargs
        tp = llm_kwargs.get("tensor-parallel-size", 1)
        pp = llm_kwargs.get("pipeline-parallel-size", 1)
        self.gpus_per_llm = tp * pp
        self.node_size = 8 if self.world_size > 1 else torch.cuda.device_count()

        place_inference_jobs = not cfg.debug.mode or cfg.debug.place_inference_workers
        if place_inference_jobs:
            self._split_gpus_by_purpose(cfg)
        else:
            self.total_finetune_gpus = self.node_size * self.world_size
            # placeholder value, wont't be used
            self.weight_update_group_size = 1

        # Place jobs on nodes in a reverse order to make sure that last node has a finetuning job going on
        self.available_gpus = {i: set(range(self.node_size)) for i in reversed(range(self.world_size))}
        self.cpu_heavy_jobs = {i: 0 for i in range(self.world_size)} 
        self.job_map = {i: [] for i in range(self.world_size)}
        self.total_jobs = 0

        if place_inference_jobs:
            self._place_inference_jobs(cfg)
        self._place_pipeline_stages(cfg)
        if cfg.environment:
            self._place_environments(cfg)

        # Place the finetune workers on the remaining gpus, take all remaining GPUs
        current_finetune_rank = 0
        finetune_rank_node = {}
        for node, remaining_gpus in self.available_gpus.items():
            gpus = list(remaining_gpus)
            if gpus:
                self.add_job(node_rank=node, kind="finetune", replica_idx=node, gpus=gpus)
                for _ in remaining_gpus:
                    finetune_rank_node[current_finetune_rank] = node
                    current_finetune_rank += 1

        assert current_finetune_rank == self.total_finetune_gpus
        if self.total_finetune_gpus % cfg.finetune.seq_parallel != 0:
            raise ValueError(
                f"Total finetune GPUs {self.total_finetune_gpus} is not divisible by seq_parallel {cfg.finetune.seq_parallel}"
            )
        for leader_idx in range(0, current_finetune_rank, cfg.finetune.seq_parallel):
            # Check that all workers in the leader's group are on the same node
            leader_node = finetune_rank_node[leader_idx]
            for offset in range(cfg.finetune.seq_parallel):
                if finetune_rank_node[leader_idx + offset] != leader_node:
                    raise ValueError(
                        f"Sequence parallel ranks {leader_idx} and {leader_idx + offset} are on different nodes: "
                        f"{finetune_rank_node[leader_idx]} and {finetune_rank_node[leader_idx + offset]}"
                    )


        # Pretty-log the world map
        self._log_info("--- WORLD MAP ---")
        for node, jobs in self.job_map.items():
            self._log_info(f"Node {node} has {len(jobs)} jobs:")
            for job in jobs:
                self._log_info(f"  {job.kind} {job.replica_idx} on gpus {job.gpus}, local idx {job.local_idx}")

    def add_job(self, node_rank: int, kind: str, replica_idx: int, local_idx: int = 0, port: int | None = None, gpus: list[int] | None = None, cpu_heavy: bool = False, url: str = "") -> Job:
        """Add a job to the world map."""
        if gpus is None:
            gpus = []
        job = Job(
            kind=kind,             
            idx=self.total_jobs,
            replica_idx=replica_idx,
            local_idx=local_idx, 
            node_rank=node_rank, 
            hostname=self.address_map[node_rank],
            port=port,
            gpus=gpus,
            url=url
        )       
        self.job_map[node_rank].append(job)
        self.total_jobs += 1
        if cpu_heavy:
            self.cpu_heavy_jobs[node_rank] += 1
        return job


    def _split_gpus_by_purpose(self, cfg):
        fraction_sum = cfg.world.actor_fraction + cfg.world.preprocessor_fraction + cfg.world.finetune_fraction
        actor_fraction = cfg.world.actor_fraction / fraction_sum
        preprocessor_fraction = cfg.world.preprocessor_fraction / fraction_sum

        # TODO: support nodes with less than 8 GPUs available
        total_gpus = self.world_size * self.node_size
        desired_actor_gpu_share = max(int(total_gpus * actor_fraction), self.gpus_per_llm)
        desired_preprocessor_gpu_share = (
            max(int(total_gpus * preprocessor_fraction), self.gpus_per_llm) if cfg.world.preprocessor_fraction else 0
        )
        desired_finetune_gpu_share = total_gpus - desired_actor_gpu_share - desired_preprocessor_gpu_share
        self._log_info(
            f"Desired GPU share: {desired_actor_gpu_share} for actors,"
            f"{desired_preprocessor_gpu_share} for preprocessors, {desired_finetune_gpu_share} for finetune"
        )

        gpus_per_actor = int(desired_actor_gpu_share / cfg.world.replicas) if cfg.world.replicas > 0 else 0
        gpus_per_actor = gpus_per_actor - (gpus_per_actor % self.gpus_per_llm)
        gpus_per_preprocessor = (
            int(desired_preprocessor_gpu_share / cfg.world.replicas) if cfg.world.replicas > 0 else 0
        )
        gpus_per_preprocessor = gpus_per_preprocessor - (gpus_per_preprocessor % self.gpus_per_llm)
        self.llms_per_actor = max(int(gpus_per_actor / self.gpus_per_llm), 1) if gpus_per_actor > 0 else 0
        self.total_actor_llms = self.llms_per_actor * cfg.world.replicas
        self.llms_per_preprocessor = (
            max(int(gpus_per_preprocessor / self.gpus_per_llm), 1) if gpus_per_preprocessor > 0 else 0
        )
        self.gpus_per_actor = gpus_per_actor
        self.gpus_per_preprocessor = gpus_per_preprocessor

        total_actor_gpus = cfg.world.replicas * gpus_per_actor
        total_preprocessor_gpus = cfg.world.replicas * gpus_per_preprocessor
        self.total_finetune_gpus = total_gpus - total_actor_gpus - total_preprocessor_gpus
        self._log_info(
            f"The configuration required:\n"
            f"{desired_actor_gpu_share} for actors, {desired_preprocessor_gpu_share} for preprocessors, {self.total_finetune_gpus} for finetune,\n"
            f"with {cfg.world.replicas} actors and {cfg.world.replicas} preprocessors,\n"
            f"and with {self.gpus_per_llm} per each LLM.\n"
        )
        self._log_info("I have adjusted the GPU shares to accomodate these constraints.")
        self._log_info(
            f"Actual GPU share: {total_actor_gpus} for actors, {total_preprocessor_gpus} for preprocessors, {self.total_finetune_gpus} for finetune"
        )
        if self.total_finetune_gpus < 0:
            raise ValueError("Not enough gpus to place all workers")
        if self.total_finetune_gpus == 0:
            logger.warning("No GPUs left for finetune workers. You can still debug other parts of the pipeline.")

        self.weight_update_group_size = self.total_actor_llms * self.gpus_per_llm + 1

    def _place_pipeline_stages(self, cfg):
        for worker_idx in range(cfg.world.replicas):
            node = self.get_least_busy_node()
            self.add_job(kind="actor", replica_idx=worker_idx, node_rank=node, gpus=[], cpu_heavy=True)
            self.add_job(kind="preprocessor", replica_idx=worker_idx, node_rank=node, gpus=[], cpu_heavy=True)

    def _place_environments(self, cfg):
        for worker_idx in range(cfg.world.env_replicas):
            node = self.get_least_busy_node()
            envs_at_node = len([job for job in self.job_map[node] if job.kind == "environment"])
            self.add_job(
                kind="environment",
                replica_idx=worker_idx,
                node_rank=node,
                port=cfg.world.environment_start_port + envs_at_node,
                gpus=[],
                cpu_heavy=True,
            )

    def _place_inference_jobs(self, cfg):
        for _ in range(cfg.world.replicas):
            for actor_llm_idx in range(self.llms_per_actor):
                node = next(
                    (node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_llm), None
                )
                if node is None:
                    raise ValueError("Not enough gpus to place all actors")
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_llm)]
                local_idx = min(gpus)
                llm_url = f"http://{self.address_map[node]}:{8080 + local_idx}"
                self.add_job(
                    kind="actor_llm",
                    replica_idx=actor_llm_idx,
                    local_idx=local_idx,
                    node_rank=node,
                    gpus=gpus,
                    port=8080 + local_idx,
                    url=llm_url,
                )

        for _ in range(cfg.world.replicas):
            for preprocessor_llm_idx in range(self.llms_per_preprocessor):
                node = next(
                    (node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_llm), None
                )
                if node is None:
                    raise ValueError("Not enough gpus to place all preprocessors")
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_llm)]
                local_idx = min(gpus)
                ref_url = f"http://{self.address_map[node]}:{8180 + local_idx}"
                self.add_job(
                    kind="preprocessor_llm",
                    replica_idx=preprocessor_llm_idx,
                    local_idx=local_idx,
                    node_rank=node,
                    gpus=gpus,
                    url=ref_url,
                )

    def get_least_busy_node(self):
        """Get the node with the least number of CPU-heavy jobs."""
        result = 0 
        for node, cpu_heavy_jobs in self.cpu_heavy_jobs.items():
            if cpu_heavy_jobs < self.cpu_heavy_jobs[result]:
                result = node
        return result

    def my_jobs(self) -> list[Job]:
        return self.job_map[self.my_rank]

    def nodes_with_finetuning(self) -> list[int]:
        return [node for node, jobs in self.job_map.items() if any(job.kind == "finetune" for job in jobs)]

    def my_finetuning_rank(self) -> int:
        return self.nodes_with_finetuning().index(self.my_rank)

    def get_all_jobs(self):
        return [job for jobs in self.job_map.values() for job in jobs]

    def get_actor_urls(self) -> list[str]:
        return [job.url for job in self.get_all_jobs() if job.kind == "actor_llm"]

    def get_preprocessor_urls(self) -> list[str]:
        return [job.url for job in self.get_all_jobs() if job.kind == "preprocessor_llm"]
