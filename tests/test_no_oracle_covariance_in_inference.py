import inspect

from wrb3d.models import WaveletResidualBridgeModel


def test_no_oracle_covariance_in_inference():
    names = set(inspect.signature(WaveletResidualBridgeModel.infer).parameters)
    assert "oracle_covariance" not in names
    assert "pet" not in names

