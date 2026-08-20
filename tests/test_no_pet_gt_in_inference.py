import inspect

from wrb3d.models import WaveletResidualBridgeModel


def test_no_pet_gt_in_inference_signature():
    names = set(inspect.signature(WaveletResidualBridgeModel.infer).parameters)
    assert not names.intersection({"pet", "gt", "target", "B"})

