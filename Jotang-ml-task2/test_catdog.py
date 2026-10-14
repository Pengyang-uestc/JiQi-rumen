"""验证训练/推理一致性、数据划分与模型保存加载，不需要重新训练。"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from catdog_model import CatDogCNN, CatDogDataset, letterbox, preprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cnn_results"


class CatDogTests(unittest.TestCase):
    def test_fully_connected_model_processes_image_batch_and_learns(self):
        from train_fc import FullyConnectedCatDog
        model = FullyConnectedCatDog()
        images = torch.randn(2, 3, 128, 128)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 2))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_letterbox_geometry_and_color(self):
        image = Image.new("RGB", (200,100), (255,0,0))
        out = np.asarray(letterbox(image))
        self.assertEqual(out.shape, (128,128,3))
        self.assertTrue(np.all(out[0] == 128))
        self.assertEqual(out[64,64].tolist(), [255,0,0])
        x = preprocess(image)
        self.assertEqual(tuple(x.shape), (3,128,128))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(-1 <= x.min() <= x.max() <= 1)

    def test_no_split_leakage(self):
        splits = json.loads((OUT / "splits.json").read_text())
        hashes = [set(r["sha256"] for r in rows) for rows in splits.values()]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
        for rows in splits.values():
            self.assertEqual({r["label"] for r in rows}, {0,1})

    def test_cache_matches_inference(self):
        rows = json.loads((OUT / "splits.json").read_text())["test"][:3]
        dataset = CatDogDataset(OUT / "images_128.npy", rows)
        for i, row in enumerate(rows):
            with Image.open(ROOT / "data" / "PetImages" / row["path"]) as image:
                expected = preprocess(image)
            self.assertTrue(torch.equal(dataset[i][0], expected))

    def test_batch_gradient_and_reload(self):
        torch.set_num_threads(2)
        model = CatDogCNN()
        x = torch.randn(2,3,128,128)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2,2))
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0,1]))
        loss.backward()
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0)
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            loaded = CatDogCNN().eval()
            loaded.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), loaded(x)))


if __name__ == "__main__":
    unittest.main()
