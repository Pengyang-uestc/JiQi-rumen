"""独立推理：python predict_catdog.py --image 图片路径 --model cnn_results/best_model.pt"""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from catdog_model import CatDogCNN, letterbox, tensor_transform


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=root / "cnn_results" / "best_model.pt")
    parser.add_argument("--output", type=Path, default=root / "cnn_results" / "prediction")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    if checkpoint["architecture"] != "CatDogCNN-v1":
        raise ValueError("Unsupported model architecture")
    model = CatDogCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with Image.open(args.image) as image:
        processed = letterbox(image, checkpoint["image_size"])
    x = tensor_transform(False)(processed).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(x).softmax(1)[0].cpu().tolist()
    index = int(probabilities[1] > probabilities[0])
    args.output.mkdir(parents=True, exist_ok=True)
    processed.save(args.output / "processed.png")
    result = {"input": str(args.image.resolve()), "prediction": ["猫", "狗"][index],
              "probability_cat": probabilities[0], "probability_dog": probabilities[1],
              "tensor_shape": list(x.shape), "device": str(device),
              "processed_image": str((args.output / "processed.png").resolve())}
    (args.output / "prediction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
