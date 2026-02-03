import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms
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

    try:
        dst.symlink_to(src)
    except Exception as e:
        raise RuntimeError(f"Failed to create symlink {dst} -> {src}.") from e


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=Path, default=Path("./"))
    parser.add_argument("--model_dir", type=Path, default=Path("."))
    parser.add_argument("--images_dir", type=Path, default=Path("./test_images"))
    parser.add_argument("--out_dir", type=Path, default=Path("./results/decoder_ablation"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_image_size", type=int, default=1024, choices=[768, 1024])
    parser.add_argument("--preprocess_size", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=1, help="0 means process all images")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\\nFree OCR. ",
    )
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

    all_images = _list_images(args.images_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found under: {args.images_dir}")

    # Pick a donor image for cross-image ablation.
    print(f"picking donor image: {all_images}")
    donor_path = all_images[0] if len(all_images) == 1 else all_images[1]
    print(f"donor image: {donor_path}")

    images = all_images
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    fill = (127, 127, 127)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # Preprocess donor once, and encode donor tokens once.
    donor_img = _square_pad_contain(Image.open(donor_path), args.preprocess_size, fill)
    donor_img_path = run_dir / "_preprocessed" / f"{donor_path.stem}.png"
    donor_img_path.parent.mkdir(parents=True, exist_ok=True)
    donor_img.save(donor_img_path)

    donor_model_img = donor_img.resize((args.model_image_size, args.model_image_size), resample=Image.BICUBIC)
    donor_tensor = tfm(donor_model_img).unsqueeze(0).to(args.device).to(torch.bfloat16)
    with torch.inference_mode():
        donor_tokens = model.model.encode_global_view(donor_tensor).squeeze(0)  # [n_query, hidden]

    summary = {
        "run_name": run_name,
        "model_dir": str(args.model_dir),
        "weights_dir": str(args.weights_dir),
        "images_dir": str(args.images_dir),
        "model_image_size": int(args.model_image_size),
        "preprocess_size": int(args.preprocess_size),
        "max_new_tokens": int(args.max_new_tokens),
        "prompt": args.prompt,
        "donor_image": str(donor_path),
        "donor_preprocessed": str(donor_img_path),
        "results": [],
    }

    def run_one(image_path: Path) -> dict:
        image_out_dir = run_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        prep = _square_pad_contain(Image.open(image_path), args.preprocess_size, fill)
        prep_path = image_out_dir / "image.png"
        prep.save(prep_path)

        def infer_with_ablation(name: str, ablation: dict | None, donor: torch.Tensor | None) -> dict:
            out_dir = image_out_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            model.model.set_causal_flow_ablation(ablation, donor)
            t0 = time.time()
            text = model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=str(prep_path),
                output_path=str(out_dir),
                base_size=args.model_image_size,
                image_size=args.model_image_size,
                crop_mode=False,  # global view only
                eval_mode=True,
                max_new_tokens=args.max_new_tokens,
            )
            dt = time.time() - t0

            out_txt = out_dir / "output.txt"
            _write_text(out_txt, text)
            return {
                "name": name,
                "output_file": str(out_txt),
                "seconds": dt,
                "num_chars": len(text),
            }

        res = {
            "image": str(image_path),
            "preprocessed": str(prep_path),
            "ablations": [],
        }

        res["ablations"].append(infer_with_ablation("baseline", None, None))
        res["ablations"].append(infer_with_ablation("replace_second_half", {"mode": "replace_second_half"}, None))
        res["ablations"].append(infer_with_ablation("swap_halves", {"mode": "swap_halves"}, None))
        res["ablations"].append(infer_with_ablation("shuffle", {"mode": "shuffle", "seed": args.seed}, None))
        res["ablations"].append(infer_with_ablation("cross_image_replace", {"mode": "cross_image_replace"}, donor_tokens))

        # Always clear ablation afterwards.
        model.model.set_causal_flow_ablation(None, None)
        return res

    for img_path in images:
        summary["results"].append(run_one(img_path))

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
