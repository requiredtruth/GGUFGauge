from pathlib import Path
import tempfile
import unittest

from ggufgauge.estimate import estimate_launch, kv_bytes_per_token
from ggufgauge.gguf import read_model_info
from ggufgauge.system import detect_capacity
from tests.fixtures import write_test_gguf


class EstimateTests(unittest.TestCase):
    def test_llama_kv_shape_and_trained_context_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = read_model_info(write_test_gguf(Path(temporary) / "model.gguf"))
            capacity = detect_capacity(ram_bytes=16 * 1024**3, logical_cpus=8)
            estimate = estimate_launch(model, capacity)
        self.assertEqual(kv_bytes_per_token(model, "f16"), 131072)
        self.assertEqual(estimate.recommended_context, 8192)
        self.assertTrue(estimate.fits_weights)

    def test_tiny_envelope_refuses_a_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = read_model_info(
                write_test_gguf(
                    Path(temporary) / "model.gguf", payload_bytes=600 * 1024**2
                )
            )
            capacity = detect_capacity(ram_bytes=512 * 1024**2, logical_cpus=2)
            estimate = estimate_launch(model, capacity)
        self.assertFalse(estimate.fits_weights)
        self.assertIsNone(estimate.recommended_context)


if __name__ == "__main__":
    unittest.main()
