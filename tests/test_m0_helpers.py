"""Cheap offline M0 checks; no GPU, model downloads, or HF login required."""
from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SmokeCodeTests(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in ("env_check.py", "configs/lora_config.py"):
            with self.subTest(rel=rel):
                ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)

    def test_required_target_names_defined_once(self):
        source = (ROOT / "configs/lora_config.py").read_text(encoding="utf-8")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            self.assertIn(f'"{name}"', source)

    def test_gitignore_protects_generated_files(self):
        contents = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for path in ("outputs/", "runs/", "data/", "*.safetensors", ".env"):
            self.assertIn(path, contents)


if __name__ == "__main__":
    unittest.main()
