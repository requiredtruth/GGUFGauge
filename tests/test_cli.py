from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from ggufgauge.cli import main
from tests.fixtures import write_test_gguf


class CliTests(unittest.TestCase):
    def test_json_output_contains_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_test_gguf(Path(temporary) / "model with space.gguf")
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        str(path),
                        "--ram-gib",
                        "16",
                        "--json",
                        "--context",
                        "4096",
                    ]
                )
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["estimate"]["recommended_context"], 8192)
        self.assertIn("'", payload["command"])
        self.assertTrue(payload["context_checks"][0]["fits"])


if __name__ == "__main__":
    unittest.main()
