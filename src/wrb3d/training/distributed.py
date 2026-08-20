"""Small DDP helpers with rank-safe metric reduction."""

from __future__ import annotations

import os
from typing import Iterator, Sized

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices across ranks without padding or duplication."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if int(num_replicas) <= 0 or not 0 <= int(rank) < int(num_replicas):
            raise ValueError("invalid distributed evaluation rank/world size")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


def initialize_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, torch.Tensor]:
    if world_size <= 1:
        return metrics
    output = {}
    for name, value in metrics.items():
        reduced = value.detach().float().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        output[name] = reduced / world_size
    return output


def gather_metric_records(records: list[dict], world_size: int) -> list[dict]:
    """Gather variable-length patient records from every rank."""
    if world_size <= 1:
        return records
    gathered: list[list[dict] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, records)
    return [row for rank_rows in gathered if rank_rows is not None for row in rank_rows]
