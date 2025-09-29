from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_onnxruntime_stub() -> None:
    if "onnxruntime" in sys.modules:
        return

    class SessionOptions:
        def __init__(self) -> None:
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

    class InferenceSession:
        def __init__(self, *args, **kwargs) -> None:
            self._inputs = [SimpleNamespace(name="input", shape=[1, 3, 512, 512])]

        def get_inputs(self):  # noqa: D401, ANN001
            return self._inputs

        def run(self, _outputs, feed_dict):  # noqa: D401, ANN001
            arr = next(iter(feed_dict.values())) if isinstance(feed_dict, dict) else None
            if isinstance(arr, np.ndarray) and arr.ndim > 0:
                batch_size = int(arr.shape[0])
            else:
                batch_size = 1
            return [np.zeros((batch_size, 100), dtype=np.float32)]

    def get_available_providers() -> list[str]:  # noqa: D401
        return ["CPUExecutionProvider"]

    stub = ModuleType("onnxruntime")
    stub.SessionOptions = SessionOptions  # type: ignore[attr-defined]
    stub.InferenceSession = InferenceSession  # type: ignore[attr-defined]
    stub.get_available_providers = get_available_providers  # type: ignore[attr-defined]
    sys.modules["onnxruntime"] = stub


def _install_nudenet_stub() -> None:
    if "nudenet" in sys.modules:
        return

    class NudeDetector:
        def __init__(self, model_name: str | None = None) -> None:  # noqa: D401
            self.model_name = model_name

        def detect_batch(self, inputs):  # noqa: D401, ANN001
            return [[] for _ in inputs]

    stub = ModuleType("nudenet")
    stub.NudeDetector = NudeDetector  # type: ignore[attr-defined]
    stub.__version__ = "stub"  # type: ignore[attr-defined]
    sys.modules["nudenet"] = stub


_install_onnxruntime_stub()
_install_nudenet_stub()
