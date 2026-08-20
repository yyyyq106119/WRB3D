import json
import platform
import sys

import torch

from wrb3d.models import WaveletResidualBridgeModel


model = WaveletResidualBridgeModel()
print(
    json.dumps(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "default_parameter_counts": model.parameter_counts(),
            "architecture_key": model.architecture_key,
        },
        indent=2,
    )
)

