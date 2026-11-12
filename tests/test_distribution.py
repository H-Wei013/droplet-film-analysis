"""Publication guards: inspect tracked source, never scan research data."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from watercar_tool import TOOL_VERSION, ALGORITHM_VERSION


class DistributionTests(unittest.TestCase):
    def test_publication_contains_source_only(self):
        git = shutil.which("git")
        if not git or not (ROOT / ".git").exists():
            self.skipTest("Requires a Git checkout and Git on PATH")
        files = subprocess.check_output([git, "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")[:-1]
        self.assertTrue(files)
        allowed_roots = {
            ".gitattributes", ".gitignore", "ENGINE_PROVENANCE.json", "README.md",
            "VALIDATION.md", "AGENTS.md", "METHODS.md", "launch.py", "pyproject.toml", "requirements.txt",
            "requirements-validated.txt", "启动工具.cmd", "配置环境.cmd",
        }
        for name in files:
            with self.subTest(file=name):
                path = Path(name)
                source = name.startswith(("src/droplet_analysis/", "src/watercar_tool/"))
                test = name.startswith("tests/")
                self.assertTrue(name in allowed_roots or source or test)
                if source:
                    self.assertTrue(path.suffix.lower() in {".py", ".js", ".html"}
                                    or name == "src/droplet_analysis/config.json")
                if test:
                    self.assertEqual(path.suffix.lower(), ".py")
                self.assertFalse({"runs", "data", "datasets", "diagnostics", "人工修正备份", "reviews", "exports", "history", "__pycache__", ".venv", ".tools"} & set(path.parts))
                self.assertFalse((ROOT / path).is_symlink())
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}", text))
                self.assertIsNone(re.search(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]", text))
                self.assertFalse(path.name.upper().startswith("LICENSE"))

    def test_ignore_rules_cover_private_artifacts(self):
        git = shutil.which("git")
        if not git or not (ROOT / ".git").exists():
            self.skipTest("Requires a Git checkout and Git on PATH")
        names = [
            "runs/a/run.json", "reviews/state.json", "exports/measurements.csv",
            "history/old.json", "local_python.cmd", "local.json", ".env",
            ".venv/pyvenv.cfg", ".tools/gh/bin/gh.exe", "nested/FRAME.BMP",
            "nested/screenshot.PNG", "nested/all.review.json", "nested/result.npz",
            "nested/measurements.json", "nested/all_frames_trace.csv",
            "diagnostics/comparison/metrics.json", "人工修正备份/notes.json",
        ]
        result = subprocess.check_output([git, "check-ignore", "--no-index", "--stdin", "-z"],
                                         input=("\0".join(names)+"\0").encode("utf-8"),
                                         cwd=ROOT).decode("utf-8").split("\0")[:-1]
        self.assertEqual(set(result), set(names))

    def test_defaults_are_independent_of_experimental_data(self):
        from droplet_analysis.core import Settings
        config = json.loads((ROOT / "src/droplet_analysis/config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["sequences"], {})
        settings = Settings.from_dict(config["defaults"])
        self.assertIsNone(settings.roi)
        self.assertIsNone(settings.wall_points)
        self.assertIsNone(settings.wall_search_roi)

    def test_release_versions_and_engine_fingerprints(self):
        proof = json.loads((ROOT / "ENGINE_PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(proof["tool_version"], TOOL_VERSION)
        self.assertEqual(proof["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(proof['baseline_algorithm_version'], '5.3.1')
        changed = {name for name, value in proof['baseline_numerical_modules_sha256'].items()
                   if value != proof['numerical_modules_sha256'][name]}
        self.assertEqual(changed, {'core.py', 'optical.py', 'v5.py'})
        self.assertEqual(changed, set(proof['modified_numerical_modules']))
        for name, expected in proof["numerical_modules_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "src/droplet_analysis" / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
