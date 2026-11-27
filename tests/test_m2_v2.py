"""M2-v2 prompt blinding tests; no model download required."""
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from make_canaries_v2 import make_entry, make_pool, TYPES


class TestM2V2(unittest.TestCase):
    def test_no_group_model_or_repeat_in_prompt(self):
        for model in ("qwen", "gemma", "llama"):
            for kind in TYPES:
                for group in ("injected", "nullgate"):
                    for index in (0, 3, 6):
                        row=make_entry(model,kind,group,index,
                                       " synthetic-target",3,9,20261120)
                        prompt=row["prompt"].casefold()
                        for forbidden in (model, group, "injected", "nullgate",
                                          "repeat", "canarysubject"):
                            self.assertNotIn(forbidden,prompt)
                        self.assertIn("syntheticperson_", prompt)

    def test_equal_metadata_blind_subjects_differ(self):
        # Changing only membership changes the random subject but does
        # not add a recognizable group label or change QA template.
        a=make_entry("qwen","id","injected",0," target",3,8,20261120)
        b=make_entry("qwen","id","nullgate",0," target",3,8,20261120)
        self.assertNotEqual(a["subject"],b["subject"])
        self.assertEqual(a["repeat"],b["repeat"])
        self.assertEqual(a["prompt"].replace(a["subject"],"X"),
                         b["prompt"].replace(b["subject"],"X"))

    def test_deterministic_v2_pool(self):
        for kind in TYPES:
            self.assertEqual(make_pool(kind,20261120,50),
                             make_pool(kind,20261120,50))
            self.assertEqual(len(set(make_pool(kind,20261120,50))),50)


if __name__=="__main__":
    unittest.main()
