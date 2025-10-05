from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for CI images
    class _Array:  # minimal stub to satisfy shape/ndim checks
        def __init__(self, shape):  # type: ignore[no-untyped-def]
            if isinstance(shape, int):
                self._shape = (shape,)
            else:
                self._shape = tuple(shape)

        @property
        def ndim(self) -> int:
            return len(self._shape)

        @property
        def shape(self) -> tuple[int, ...]:
            return self._shape


    def _zeros(shape, dtype=None):  # type: ignore[no-untyped-def]
        return _Array(shape)


    np = ModuleType("numpy")
    np.ndarray = _Array  # type: ignore[attr-defined]
    np.zeros = _zeros  # type: ignore[attr-defined]
    np.float32 = float  # type: ignore[attr-defined]
    sys.modules["numpy"] = np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.profiles import clear_context_cache, set_profiles_root_override


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


@pytest.fixture
def profiles_root_override(tmp_path: Path):
    """Temporarily redirect profile outputs to a tmp directory."""

    set_profiles_root_override(tmp_path)
    try:
        yield tmp_path
    finally:
        set_profiles_root_override(None)
        clear_context_cache()
