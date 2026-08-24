from pathlib import Path
import tempfile
import unittest

from ggufgauge.system import detect_capacity


class SystemCapacityTests(unittest.TestCase):
    def test_cgroup_remaining_memory_caps_host_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meminfo = root / "meminfo"
            meminfo.write_text("MemAvailable: 16777216 kB\n", encoding="ascii")
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.max").write_text(str(8 * 1024**3), encoding="ascii")
            (cgroup / "memory.current").write_text(
                str(2 * 1024**3), encoding="ascii"
            )
            capacity = detect_capacity(
                proc_meminfo=meminfo, cgroup_root=cgroup, logical_cpus=6
            )
        self.assertEqual(capacity.available_bytes, 6 * 1024**3)
        self.assertEqual(capacity.memory_source, "remaining cgroup memory")
        self.assertEqual(capacity.recommended_threads, 5)

    def test_operator_override_is_exact(self) -> None:
        capacity = detect_capacity(ram_bytes=12 * 1024**3, logical_cpus=4)
        self.assertEqual(capacity.available_bytes, 12 * 1024**3)
        self.assertEqual(capacity.memory_source, "operator override")


if __name__ == "__main__":
    unittest.main()
