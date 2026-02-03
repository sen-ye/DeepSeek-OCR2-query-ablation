import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageOps
import torch
from transformers import AutoModel, AutoTokenizer


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_weight_file(model_dir: Path, weights_dir: Path) -> None:
    model_dir = model_dir.resolve()
    weights_dir = weights_dir.resolve()

    weight_name = "model-00001-of-000001.safetensors"
    dst = model_dir / weight_name
    if dst.exists():
        return

    src = weights_dir / weight_name
    if not src.exists():
        raise FileNotFoundError(f"Missing weights file: {src}")

    dst.symlink_to(src)


def _list_images(images_dir: Path) -> list[Path]:
    images = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def _square_pad_contain(img: Image.Image, size: int, fill: tuple[int, int, int]) -> Image.Image:
    img = ImageOps.exif_transpose(img.convert("RGB"))
    resized = ImageOps.contain(img, (size, size))
    canvas = Image.new("RGB", (size, size), fill)
    x = (size - resized.size[0]) // 2
    y = (size - resized.size[1]) // 2
    canvas.paste(resized, (x, y))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=Path, default=Path("./"))
    parser.add_argument("--model_dir", type=Path, default=Path("."))
    parser.add_argument("--images_dir", type=Path, default=Path("./test_images"))
    parser.add_argument("--out_dir", type=Path, default=Path("./results/free_ocr"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--preprocess_size", type=int, default=512)
    parser.add_argument("--model_image_size", type=int, default=1024, choices=[768, 1024])
    parser.add_argument("--max_images", type=int, default=0, help="0 means process all images")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--prompt", type=str, default="<image>\\nFree OCR.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_weight_file(args.model_dir, args.weights_dir)

    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="flash_attention_2",
    )
    model = model.eval().to(args.device).to(torch.bfloat16)

    img_paths = _list_images(args.images_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {args.images_dir}")
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    fill = (127, 127, 127)
    results = []
    for img_path in img_paths:
        img_out_dir = run_dir / img_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path)
        if args.preprocess_size and args.preprocess_size > 0:
            img = _square_pad_contain(img, args.preprocess_size, fill)

        img_save = img_out_dir / "image.png"
        img.save(img_save)

        t0 = time.time()
        text = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(img_save),
            output_path=str(img_out_dir),
            base_size=args.model_image_size,
            image_size=args.model_image_size,
            crop_mode=False,
            eval_mode=True,
            max_new_tokens=args.max_new_tokens,
        )
        dt = time.time() - t0

        out_txt = img_out_dir / "output.txt"
        out_txt.write_text(text, encoding="utf-8")
        results.append(
            {
                "image": str(img_path),
                "preprocessed": str(img_save),
                "output_file": str(out_txt),
                "seconds": dt,
                "num_chars": len(text),
            }
        )

    summary = {
        "run_name": run_name,
        "prompt": args.prompt,
        "model_image_size": int(args.model_image_size),
        "preprocess_size": int(args.preprocess_size),
        "max_new_tokens": int(args.max_new_tokens),
        "images_dir": str(args.images_dir),
        "num_images": len(img_paths),
        "results": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
