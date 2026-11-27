"""M3 CPU-only unit tests; requires torch but not model downloads."""
import tempfile
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from utils import encode_qa, tensor_batch, batch_order


class FakeTokenizer:
    eos_token="<EOS>"
    pad_token_id=0
    def __call__(self, text, add_special_tokens=True):
        ids=[1] if add_special_tokens else []
        ids.extend([ord(x)+4 for x in text])
        return {"input_ids":ids}


class M3Tests(unittest.TestCase):
    def test_answer_only_mask_and_eos(self):
        tok=FakeTokenizer()
        ids,labels,n=encode_qa(tok,"Question: test?\nAnswer:"," synthetic",512)
        self.assertTrue(all(x==-100 for x in labels[:-n]))
        self.assertTrue(all(x!=-100 for x in labels[-n:]))
        self.assertEqual(ids[-n:],labels[-n:])
        self.assertGreater(n,1)

    def test_batch_padding_masks(self):
        tok=FakeTokenizer()
        rows=[encode_qa(tok,"Question: A\nAnswer:"," x",512),
              encode_qa(tok,"Question: longer\nAnswer:"," y",512)]
        batch=tensor_batch(rows,0,"cpu")
        self.assertEqual(batch["input_ids"].shape,batch["labels"].shape)
        self.assertTrue((batch["labels"][batch["attention_mask"]==0]==-100).all())

    def test_shuffle_stability(self):
        self.assertEqual(batch_order(20,42,0),batch_order(20,42,0))
        self.assertNotEqual(batch_order(20,42,0),batch_order(20,43,0))
        self.assertEqual(sorted(batch_order(20,42,0)),list(range(20)))


if __name__=="__main__":
    unittest.main()
