from pathlib import Path
import tempfile
import unittest

from ggufgauge.gguf import GGUFError, read_model_info
from tests.fixtures import write_test_gguf


class GGUFReaderTests(unittest.TestCase):
    def test_reads_required_architecture_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_test_gguf(Path(temporary) / "model.gguf")
            model = read_model_info(path)
        self.assertEqual(model.version, 3)
        self.assertEqual(model.architecture, "llama")
        self.assertEqual(model.block_count, 32)
        self.assertEqual(model.head_count_kv, 8)
        self.assertEqual(model.context_length, 8192)

    def test_rejects_non_gguf_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.gguf"
            path.write_bytes(b"not a gguf file at all!!!!")
            with self.assertRaises(GGUFError):
                read_model_info(path)


if __name__ == "__main__":
    unittest.main()
