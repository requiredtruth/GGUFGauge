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
                proc_meminfo=meminfo,
                proc_self_cgroup=root / "missing-self-cgroup",
                cgroup_root=cgroup,
                logical_cpus=6,
            )
        self.assertEqual(capacity.available_bytes, 6 * 1024**3)
        self.assertEqual(capacity.memory_source, "remaining cgroup memory")
        self.assertEqual(capacity.recommended_threads, 5)
        self.assertEqual(capacity.cgroup_path, "/")

    def test_nested_v2_process_cgroup_caps_root_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meminfo = root / "meminfo"
            meminfo.write_text("MemAvailable: 33554432 kB\n", encoding="ascii")
            proc_cgroup = root / "self-cgroup"
            proc_cgroup.write_text("0::/user.slice/job.scope\n", encoding="ascii")
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.max").write_text(str(16 * 1024**3), encoding="ascii")
            (cgroup / "memory.current").write_text(str(1 * 1024**3), encoding="ascii")
            nested = cgroup / "user.slice" / "job.scope"
            nested.mkdir(parents=True)
            (nested / "memory.max").write_text(str(4 * 1024**3), encoding="ascii")
            (nested / "memory.current").write_text(str(1 * 1024**3), encoding="ascii")
            capacity = detect_capacity(
                proc_meminfo=meminfo,
                proc_self_cgroup=proc_cgroup,
                cgroup_root=cgroup,
                logical_cpus=4,
            )
        self.assertEqual(capacity.available_bytes, 3 * 1024**3)
        self.assertEqual(capacity.cgroup_remaining_bytes, 3 * 1024**3)
        self.assertEqual(capacity.cgroup_path, "/user.slice/job.scope")

    def test_parent_v2_limit_can_be_tighter_than_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meminfo = root / "meminfo"
            meminfo.write_text("MemAvailable: 33554432 kB\n", encoding="ascii")
            proc_cgroup = root / "self-cgroup"
            proc_cgroup.write_text("0::/workload/leaf\n", encoding="ascii")
            cgroup = root / "cgroup"
            parent = cgroup / "workload"
            leaf = parent / "leaf"
            leaf.mkdir(parents=True)
            (parent / "memory.max").write_text(str(6 * 1024**3), encoding="ascii")
            (parent / "memory.current").write_text(str(4 * 1024**3), encoding="ascii")
            (leaf / "memory.max").write_text(str(8 * 1024**3), encoding="ascii")
            (leaf / "memory.current").write_text(str(1 * 1024**3), encoding="ascii")
            capacity = detect_capacity(
                proc_meminfo=meminfo,
                proc_self_cgroup=proc_cgroup,
                cgroup_root=cgroup,
                logical_cpus=4,
            )
        self.assertEqual(capacity.available_bytes, 2 * 1024**3)
        self.assertEqual(capacity.cgroup_path, "/workload")

    def test_unsafe_v2_path_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meminfo = root / "meminfo"
            meminfo.write_text("MemAvailable: 16777216 kB\n", encoding="ascii")
            proc_cgroup = root / "self-cgroup"
            proc_cgroup.write_text("0::/../../outside\n", encoding="ascii")
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.max").write_text(str(8 * 1024**3), encoding="ascii")
            (cgroup / "memory.current").write_text(str(2 * 1024**3), encoding="ascii")
            capacity = detect_capacity(
                proc_meminfo=meminfo,
                proc_self_cgroup=proc_cgroup,
                cgroup_root=cgroup,
                logical_cpus=4,
            )
        self.assertEqual(capacity.available_bytes, 6 * 1024**3)
        self.assertEqual(capacity.cgroup_path, "/")

    def test_operator_override_is_exact(self) -> None:
        capacity = detect_capacity(ram_bytes=12 * 1024**3, logical_cpus=4)
        self.assertEqual(capacity.available_bytes, 12 * 1024**3)
        self.assertEqual(capacity.memory_source, "operator override")


if __name__ == "__main__":
    unittest.main()
