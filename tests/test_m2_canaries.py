"""M2 offline regression tests (no tokenizer/model downloads)."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from make_canaries import answer_for, prompt_for, make_pool, pick_equal_length, make_entry


class FakeTokenizer:
    """Variable-length stand-in to test the selection algorithm only."""
    def __call__(self, items, add_special_tokens=False):
        if isinstance(items, str):
            items = [items]
        # Treat chars in groups of 4 as artificial tokens.
        return {"input_ids": [[0]*((len(text)+3)//4) for text in items]}


class CanaryTests(unittest.TestCase):
    def test_pool_reproducible_and_unique(self):
        for kind in ("id", "phone", "name_dx"):
            a = make_pool(kind, 20261120, 150)
            b = make_pool(kind, 20261120, 150)
            self.assertEqual(a, b)
            self.assertEqual(len(set(a)), 150)

    def test_phone_uses_fictional_range(self):
        for text in make_pool("phone", 3, 100):
            self.assertIn("-555-01", text)
            self.assertTrue(text.startswith(" +1-"))

    def test_exact_length_bucket(self):
        t = FakeTokenizer()
        pool = make_pool("id", 42, 200)
        length, chosen, distribution = pick_equal_length(pool, t, 25, 42, "id")
        self.assertEqual(len(chosen), 25)
        self.assertTrue(all(len(t(x)["input_ids"][0]) == length for x in chosen))
        self.assertGreaterEqual(sum(distribution.values()), 200)

    def test_repetitions_and_subjects(self):
        for i, expected_repeat in ((0, 1), (3, 5), (6, 20)):
            c = make_entry("qwen", "name_dx", "injected", i,
                           " Synthetic diagnosis DX-TEST", 3, 10)
            self.assertEqual(c["repeat"], expected_repeat)
            self.assertTrue(c["prompt"].startswith("Question: "))
            self.assertIn("CanarySubject_", c["prompt"])


if __name__ == "__main__":
    unittest.main()
