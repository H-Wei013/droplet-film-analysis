"""Publication guards: inspect tracked source, never scan research data."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_publication_contains_source_only(self):
        git = shutil.which("git")
        if not git or not (ROOT / ".git").exists():
            self.skipTest("Requires a Git checkout and Git on PATH")
        files = subprocess.check_output([git, "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")[:-1]
        self.assertTrue(files)
        allowed_roots = {
            ".gitattributes", ".gitignore", "ENGINE_PROVENANCE.json", "README.md",
            "VALIDATION.md", "launch.py", "pyproject.toml", "requirements.txt",
            "requirements-validated.txt", "启动工具.cmd", "配置环境.cmd",
        }
        for name in files:
            with self.subTest(file=name):
                path = Path(name)
                self.assertTrue(name in allowed_roots or name.startswith(("src/", "tests/")))
                self.assertIn(path.suffix.lower(), {"", ".py", ".js", ".html", ".json", ".md", ".toml", ".txt", ".cmd"})
                self.assertFalse({"runs", "data", "datasets", "reviews", "exports", "history", "__pycache__", ".venv", ".tools"} & set(path.parts))
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
        ]
        result = subprocess.check_output([git, "check-ignore", "--no-index", "--stdin"],
                                         input=("\n".join(names)+"\n").encode(), cwd=ROOT).decode().splitlines()
        self.assertEqual(set(result), set(names))

    def test_release_versions_and_engine_fingerprints(self):
        proof = json.loads((ROOT / "ENGINE_PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(proof["tool_version"], "0.1.0")
        self.assertEqual(proof["algorithm_version"], "5.3.1")
        for name, expected in proof["unchanged_numerical_modules_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "src/droplet_analysis" / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
