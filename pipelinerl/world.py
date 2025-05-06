import logging
import os
from pydantic import BaseModel
from omegaconf import DictConfig
import torch
logger = logging.getLogger(__name__)

class Job(BaseModel):
    """Represent the decision to launch a replica of a particular worker (e.g. actor) at a particular rank"""
    kind: str
    # The global index of this job among jobs of the same kind
    replica_idx: int
    # The index of this job among similar jobs on the same node
    local_idx: int = 0
    # Where this job should run
    node_rank: int
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
                basename = self.master_addr[:self.master_addr.rfind("-")]
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

        if not cfg.debug.mode == "finetune" or cfg.debug.place_inference_workers:
            self._split_gpus_by_purpose(cfg)
        else:
            self.total_finetune_gpus = self.node_size * self.world_size
            # placeholder value, wont't be used
            self.weight_update_group_size = 1

        # Place jobs on nodes in a reverse order to make sure that last node has a finetuning job going on
        self.available_gpus: dict[int, set] = {i: set(range(self.node_size)) for i in reversed(range(self.world_size))}
        self.job_map = {i: [] for i in range(self.world_size)}

        if not cfg.debug.mode == "finetune" or cfg.debug.place_inference_workers:
            self._place_inference_jobs(cfg)

        # Place the finetune workers on the remaining gpus, take all remaining GPUs
        for node, remaining_gpus in self.available_gpus.items():
            gpus = list(remaining_gpus)
            if gpus:
                self.job_map[node].append(Job(kind="finetune", replica_idx=node, node_rank=node, gpus=gpus))

        # Pretty-log the world map
        self._log_info("--- WORLD MAP ---")
        for node, jobs in self.job_map.items():
            self._log_info(f"Node {node} has {len(jobs)} jobs:")
            for job in jobs:
                self._log_info(f"  {job.kind} {job.replica_idx} on gpus {job.gpus}, local idx {job.local_idx}")

    def _split_gpus_by_purpose(self, cfg):
        fraction_sum = cfg.world.actor_fraction + cfg.world.preprocessor_fraction + cfg.world.finetune_fraction
        actor_fraction = cfg.world.actor_fraction / fraction_sum
        preprocessor_fraction = cfg.world.preprocessor_fraction / fraction_sum

        # TODO: support nodes with less than 8 GPUs available
        total_gpus = self.world_size * self.node_size
        desired_actor_gpu_share = max(int(total_gpus * actor_fraction), self.gpus_per_llm)
        desired_preprocessor_gpu_share = (
            max(int(total_gpus * preprocessor_fraction), self.gpus_per_llm)
            if cfg.world.preprocessor_fraction
            else 0
        )
        desired_finetune_gpu_share = total_gpus - desired_actor_gpu_share - desired_preprocessor_gpu_share
        self._log_info(
            f"Desired GPU share: {desired_actor_gpu_share} for actors,"
            f"{desired_preprocessor_gpu_share} for preprocessors, {desired_finetune_gpu_share} for finetune"
        )

        gpus_per_actor = int(desired_actor_gpu_share / cfg.world.actors) if cfg.world.actors > 0 else 0
        gpus_per_actor = gpus_per_actor - (gpus_per_actor % self.gpus_per_llm)
        gpus_per_preprocessor = int(desired_preprocessor_gpu_share / cfg.world.preprocessors) if cfg.world.preprocessors > 0 else 0
        gpus_per_preprocessor = gpus_per_preprocessor - (gpus_per_preprocessor % self.gpus_per_llm)
        self.llms_per_actor = max(int(gpus_per_actor / self.gpus_per_llm), 1) if gpus_per_actor > 0 else 0
        self.total_actor_llms = self.llms_per_actor * cfg.world.actors
        self.llms_per_preprocessor = max(int(gpus_per_preprocessor / self.gpus_per_llm), 1) if gpus_per_preprocessor > 0 else 0
        self.gpus_per_actor = gpus_per_actor
        self.gpus_per_preprocessor = gpus_per_preprocessor

        total_actor_gpus = cfg.world.actors * gpus_per_actor
        total_preprocessor_gpus = cfg.world.preprocessors * gpus_per_preprocessor
        self.total_finetune_gpus = total_gpus - total_actor_gpus - total_preprocessor_gpus
        self._log_info(
            f"The configuration required:\n"
            f"{desired_actor_gpu_share} for actors, {desired_preprocessor_gpu_share} for preprocessors, {self.total_finetune_gpus} for finetune,\n"
            f"with {cfg.world.actors} actors and {cfg.world.preprocessors} preprocessors,\n"
            f"and with {self.gpus_per_llm} per each LLM.\n"
        )
        self._log_info("I have adjusted the GPU shares to accomodate these constraints.")
        self. _log_info(f"Actual GPU share: {total_actor_gpus} for actors, {total_preprocessor_gpus} for preprocessors, {self.total_finetune_gpus} for finetune")
        if self.total_finetune_gpus < 0:
            raise ValueError("Not enough gpus to place all workers")
        if self.total_finetune_gpus == 0:
            logger.warning("No GPUs left for finetune workers. You can still debug other parts of the pipeline.")

        self.weight_update_group_size = self.total_actor_llms * self.gpus_per_llm + 1

    def _place_inference_jobs(self, cfg):
        actor_placed = False
        for worker_idx in range(cfg.world.actors):
            for actor_llm_idx in range(self.llms_per_actor):
                node = next((node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_llm), None)
                if node is None:
                    raise ValueError("Not enough gpus to place all actors")
                if not actor_placed:
                    self.job_map[node].append(
                        Job(kind="actor", replica_idx=worker_idx, node_rank=node, gpus=[])
                    )
                    self.job_map[node].append(
                        Job(kind="verifier", replica_idx=worker_idx, node_rank=node, gpus=[])
                    )
                    actor_placed = True
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_llm)]
                local_idx = min(gpus)
                llm_url = f"http://{self.address_map[node]}:{8080 + local_idx}"
                self.job_map[node].append(
                    Job(
                        kind="actor_llm", replica_idx=actor_llm_idx, 
                        local_idx=local_idx, node_rank=node, gpus=gpus, url=llm_url
                    )
                )

        preprocessor_placed = False
        for worker_idx in range(cfg.world.preprocessors):
            for preprocessor_llm_idx in range(self.llms_per_preprocessor):
                node = next((node for node in self.available_gpus if len(self.available_gpus[node]) >= self.gpus_per_llm), None)
                if node is None:
                    raise ValueError("Not enough gpus to place all preprocessors")
                if not preprocessor_placed:
                    self.job_map[node].append(
                        Job(kind="preprocessor", replica_idx=worker_idx, node_rank=node, gpus=[])
                    )
                    preprocessor_placed = True
                gpus = [self.available_gpus[node].pop() for _ in range(self.gpus_per_llm)]
                local_idx = min(gpus)
                ref_url = f"http://{self.address_map[node]}:{8180 + local_idx}"
                self.job_map[node].append(
                    Job(
                        kind="preprocessor_llm", replica_idx=preprocessor_llm_idx, 
                        local_idx=local_idx, node_rank=node, gpus=gpus, url=ref_url
                    )
                )
        if not preprocessor_placed:
            assert cfg.world.preprocessor_fraction == 0
            self.job_map[self.world_size - 1].append(
                Job(kind="preprocessor", replica_idx=0, node_rank=self.world_size - 1, gpus=[])
            )


    def my_jobs(self) -> list[Job]:
        return self.job_map[self.my_rank]
    
    def nodes_with_finetuning(self) -> list[int]:
        return [node for node, jobs in self.job_map.items() if any(job.kind == "finetune" for job in jobs)]
    
    def my_finetuning_rank(self) -> int:
        return self.nodes_with_finetuning().index(self.my_rank)

    def get_all_jobs(self):
        return [job for jobs in self.job_map.values() for job in jobs]

    def get_actor_urls(self) -> list[str]:
        return [
            job.url for job in self.get_all_jobs() if job.kind == "actor_llm"
        ]
    
    def get_preprocessor_urls(self) -> list[str]:
        return [
            job.url for job in self.get_all_jobs() if job.kind == "preprocessor_llm"
        ]