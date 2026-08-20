from wrb3d.training import DistributedEvalSampler


def test_distributed_eval_sampler_has_exact_unique_coverage():
    dataset = list(range(7))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=4, rank=rank))
        for rank in range(4)
    ]
    combined = [index for shard in shards for index in shard]
    assert sorted(combined) == list(range(7))
    assert len(combined) == len(set(combined))
