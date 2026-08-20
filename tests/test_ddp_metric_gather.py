import torch

from wrb3d.training.distributed import reduce_metrics


def test_ddp_metric_gather_single_process_identity():
    metrics = {"loss": torch.tensor(2.0), "mae": torch.tensor(0.4)}
    output = reduce_metrics(metrics, 1)
    assert output is metrics

