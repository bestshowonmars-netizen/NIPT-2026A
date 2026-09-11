"""Regression checks for archived Windows evidence in a relocated checkout."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap
import hashlib
import json
import tempfile
import unittest
from validation.evidence_paths import project_path

ROOT = bootstrap.ROOT


class EvidencePathTests(unittest.TestCase):
    def test_actual_source_manifests_on_both_path_flavours(self):
        for question in (3, 4):
            meta = json.loads((ROOT / f"q{question}/solution.json").read_text(encoding="utf-8"))
            for recorded, expected in meta["source_hashes"].items():
                actual = project_path(recorded, ROOT)
                self.assertTrue(actual.is_relative_to(ROOT))
                self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(), expected)
                relative = actual.relative_to(ROOT).as_posix()
                for moved in (PurePosixPath("/srv/team/renamed_checkout"),
                              PureWindowsPath("Z:/team/renamed_checkout")):
                    self.assertEqual(project_path(recorded, moved), moved / relative)

    def test_old_checkout_existing_does_not_override_current_copy(self):
        recorded = ROOT / "data/attachment1.xlsx"
        self.assertTrue(recorded.is_file())
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / "renamed_checkout"
            mapped = project_path(recorded, moved)
            self.assertEqual(mapped, moved / "data/attachment1.xlsx")
            self.assertFalse(mapped.exists())
            # A missing copied artifact remains missing despite the original.
            with self.assertRaises(FileNotFoundError):
                mapped.read_bytes()

    def test_relative_windows_separators_are_portable(self):
        moved = PurePosixPath("/srv/team/project")
        self.assertEqual(project_path(r"q3\runs\example.npz", moved),
                         moved / "q3/runs/example.npz")

    def test_windows_drive_and_unc_project_records(self):
        moved = PurePosixPath("/srv/team/project")
        for recorded in (r"E:\old\2026A-herbal-drying\q2\solution.npz",
                         r"\\server\share\2026A-HERBAL-DRYING\q2\solution.npz"):
            self.assertEqual(project_path(recorded, moved), moved / "q2/solution.npz")

    def test_posix_project_records_on_windows(self):
        moved = PureWindowsPath("Z:/team/project")
        self.assertEqual(project_path("/old/2026A-herbal-drying/q3/solution.npz", moved),
                         moved / "q3/solution.npz")

    def test_unrelated_native_absolute_path_remains_external(self):
        for root, external in ((PurePosixPath("/srv/team/project"), "/data/external.npz"),
                               (PureWindowsPath("Z:/team/project"), "E:/external.npz")):
            self.assertEqual(project_path(external, root), type(root)(external))

    def test_unrelated_foreign_absolute_path_is_not_guessed(self):
        for root, external in ((PurePosixPath("/srv/team/project"), "E:/external.npz"),
                               (PureWindowsPath("Z:/team/project"), "/data/external.npz")):
            with self.assertRaises(ValueError):
                project_path(external, root)


if __name__ == "__main__":
    unittest.main()
