import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms
from transformers import AutoModel

import deepencoderv2


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


def _round_floats(values: list[float], ndigits: int = 6) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _extract_query_weights(
    *,
    attn_mean: torch.Tensor,  # [seq, seq], already averaged over heads/layers
    n_query: int,
) -> list[list[float]]:
    grid_size = int(math.isqrt(n_query))
    if grid_size * grid_size != n_query:
        raise ValueError(f"n_query must be a perfect square for visualization: n_query={n_query}")

    weights: list[list[float]] = []
    for qid in range(n_query):
        tgt = n_query + qid
        w = attn_mean[tgt, :n_query].clamp(min=0)
        w_sum = w.sum()
        if w_sum.item() <= 0:
            w = torch.full_like(w, 1.0 / n_query)
        else:
            w = w / w_sum
        weights.append(_round_floats(w.detach().cpu().tolist(), ndigits=6))

    return weights


HTML_TEMPLATE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DeepSeek-OCR-2 Query Attention</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }
      .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
      .layers { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
      #canvas { border: 1px solid #ddd; image-rendering: pixelated; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
      select, input[type="range"] { width: 280px; }
      button { padding: 6px 10px; }
    </style>
  </head>
  <body>
    <h2>DeepSeek-OCR-2 Query Token Attention (encoder last __LAST_N_LAYERS__ layers)</h2>

    <div class="row">
      <div>
        <div><b>Image</b>: <span class="mono">__IMAGE_NAME__</span></div>
        <div><b>n_query</b>: <span class="mono" id="nq"></span>, <b>grid</b>: <span class="mono" id="gs"></span></div>
        <div><b>Layers</b>: <span class="mono" id="layers_sel"></span></div>
        <div><b>Centroid</b>: <span class="mono" id="centroid"></span></div>
        <div><b>BBox</b>: <span class="mono" id="bbox"></span></div>
      </div>
      <div>
        <div><label>Query ID: <span class="mono" id="qid_label"></span></label></div>
        <input id="qid" type="range" min="0" step="1" value="0" />
        <div style="height: 6px;"></div>
        <select id="qid_select"></select>
        <div style="height: 10px;"></div>
        <div><label>Heatmap alpha: <span class="mono" id="alpha_label"></span></label></div>
        <input id="alpha" type="range" min="0" max="100" step="1" value="75" />
      </div>
    </div>

    <div style="height: 10px;"></div>
    <div>
      <div><b>Layer selection</b> (mean over selected):</div>
      <div id="layers" class="layers"></div>
      <div style="height: 6px;"></div>
      <button id="layers_all" type="button">Select all</button>
      <button id="layers_none" type="button">Clear</button>
    </div>

    <div style="height: 12px;"></div>
    <canvas id="canvas" width="__DISPLAY_SIZE__" height="__DISPLAY_SIZE__"></canvas>

    <script src="data.js"></script>
    <script>
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");

      const qid = document.getElementById("qid");
      const qidSelect = document.getElementById("qid_select");
      const qidLabel = document.getElementById("qid_label");
      const alpha = document.getElementById("alpha");
      const alphaLabel = document.getElementById("alpha_label");

      const nqEl = document.getElementById("nq");
      const gsEl = document.getElementById("gs");
      const layersSelEl = document.getElementById("layers_sel");
      const centroidEl = document.getElementById("centroid");
      const bboxEl = document.getElementById("bbox");

      const layersDiv = document.getElementById("layers");
      const layersAllBtn = document.getElementById("layers_all");
      const layersNoneBtn = document.getElementById("layers_none");

      function fmtBox(b) {
        return `(${b.x1.toFixed(1)}, ${b.y1.toFixed(1)}) - (${b.x2.toFixed(1)}, ${b.y2.toFixed(1)})`;
      }

      function computeStats(weights, gridSize, massThreshold) {
        const n = weights.length;
        let sum = 0.0;
        for (let i = 0; i < n; i++) sum += weights[i];
        if (!isFinite(sum) || sum <= 0) {
          for (let i = 0; i < n; i++) weights[i] = 1.0 / n;
          sum = 1.0;
        } else {
          for (let i = 0; i < n; i++) weights[i] = weights[i] / sum;
        }

        const cell = canvas.width / gridSize;
        let cx = 0.0, cy = 0.0;
        for (let r = 0; r < gridSize; r++) {
          for (let c = 0; c < gridSize; c++) {
            const w = weights[r * gridSize + c];
            cx += w * (c + 0.5) * cell;
            cy += w * (r + 0.5) * cell;
          }
        }

        const idx = Array.from({ length: n }, (_, i) => i);
        idx.sort((a, b) => weights[b] - weights[a]);
        let cum = 0.0;
        let r0 = gridSize, r1 = 0, c0 = gridSize, c1 = 0;
        for (let t = 0; t < idx.length; t++) {
          const i = idx[t];
          cum += weights[i];
          const r = Math.floor(i / gridSize);
          const c = i % gridSize;
          r0 = Math.min(r0, r);
          r1 = Math.max(r1, r);
          c0 = Math.min(c0, c);
          c1 = Math.max(c1, c);
          if (cum >= massThreshold) break;
        }

        const bbox = {
          x1: c0 * cell,
          y1: r0 * cell,
          x2: (c1 + 1) * cell,
          y2: (r1 + 1) * cell,
        };

        return {
          weights,
          centroid: { x: cx, y: cy },
          bbox,
        };
      }

      function getSelectedLayerPositions() {
        const checks = Array.from(layersDiv.querySelectorAll("input[type=checkbox]"));
        const selected = checks.filter(cb => cb.checked).map(cb => Number(cb.dataset.pos));
        if (selected.length > 0) return selected;
        return checks.map(cb => Number(cb.dataset.pos));
      }

      function setAllLayers(checked) {
        const checks = Array.from(layersDiv.querySelectorAll("input[type=checkbox]"));
        for (const cb of checks) cb.checked = checked;
      }

      function getAveragedWeights(data, qidValue, layerPositions) {
        const n = data.n_query;
        const out = new Array(n).fill(0.0);
        for (const lp of layerPositions) {
          const w = data.layer_query_weights[lp][qidValue];
          for (let i = 0; i < n; i++) out[i] += w[i];
        }
        const inv = 1.0 / layerPositions.length;
        for (let i = 0; i < n; i++) out[i] *= inv;
        return out;
      }

      async function main() {
        const data = window.ENCODER_QUERY_ATTN_DATA;
        const img = new Image();
        img.src = "image.png";
        await img.decode();

        const nQuery = data.n_query;
        const gridSize = data.grid_size;
        qid.max = String(nQuery - 1);
        nqEl.textContent = String(nQuery);
        gsEl.textContent = `${gridSize}x${gridSize}`;

        // Layer checkboxes
        layersDiv.innerHTML = "";
        for (let i = 0; i < data.layer_indices.length; i++) {
          const li = data.layer_indices[i];
          const wrap = document.createElement("label");
          wrap.className = "mono";
          wrap.style.display = "inline-flex";
          wrap.style.alignItems = "center";
          wrap.style.gap = "6px";

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.dataset.pos = String(i);
          cb.checked = true;
          cb.addEventListener("change", () => setQuery(Number(qid.value)));

          const txt = document.createElement("span");
          txt.textContent = `L${li}`;

          wrap.appendChild(cb);
          wrap.appendChild(txt);
          layersDiv.appendChild(wrap);
        }

        layersAllBtn.addEventListener("click", () => {
          setAllLayers(true);
          setQuery(Number(qid.value));
        });
        layersNoneBtn.addEventListener("click", () => {
          setAllLayers(false);
          setQuery(Number(qid.value));
        });

        qidSelect.innerHTML = "";
        for (let i = 0; i < nQuery; i++) {
          const opt = document.createElement("option");
          opt.value = String(i);
          opt.textContent = String(i);
          qidSelect.appendChild(opt);
        }

        function draw(qidValue) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          const layerPos = getSelectedLayerPositions();
          const layerIdx = layerPos.map(p => data.layer_indices[p]);
          layersSelEl.textContent = layerIdx.length ? layerIdx.join(",") : "all";

          const weights = getAveragedWeights(data, qidValue, layerPos);
          const stats = computeStats(weights, gridSize, data.mass_threshold);

          let maxW = 0.0;
          for (let i = 0; i < stats.weights.length; i++) maxW = Math.max(maxW, stats.weights[i]);
          if (maxW <= 0) maxW = 1.0;

          const cell = canvas.width / gridSize;
          const a = Number(alpha.value) / 100.0;
          alphaLabel.textContent = a.toFixed(2);

          for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
              const v = stats.weights[r * gridSize + c] / maxW;
              if (v <= 0) continue;
              ctx.fillStyle = `rgba(255,0,0,${Math.min(0.95, v * a)})`;
              ctx.fillRect(c * cell, r * cell, cell, cell);
            }
          }

          // bbox
          ctx.strokeStyle = "lime";
          ctx.lineWidth = 2;
          ctx.strokeRect(stats.bbox.x1, stats.bbox.y1, stats.bbox.x2 - stats.bbox.x1, stats.bbox.y2 - stats.bbox.y1);

          // centroid
          ctx.fillStyle = "cyan";
          ctx.beginPath();
          ctx.arc(stats.centroid.x, stats.centroid.y, 3.5, 0, 2 * Math.PI);
          ctx.fill();

          centroidEl.textContent = `(${stats.centroid.x.toFixed(1)}, ${stats.centroid.y.toFixed(1)})`;
          bboxEl.textContent = fmtBox(stats.bbox);
        }

        function setQuery(i) {
          i = Math.max(0, Math.min(nQuery - 1, i));
          qid.value = String(i);
          qidSelect.value = String(i);
          qidLabel.textContent = String(i);
          draw(i);
        }

        qid.addEventListener("input", () => setQuery(Number(qid.value)));
        qidSelect.addEventListener("change", () => setQuery(Number(qidSelect.value)));
        alpha.addEventListener("input", () => setQuery(Number(qid.value)));

        document.addEventListener("keydown", (e) => {
          if (e.key === "ArrowLeft") setQuery(Number(qid.value) - 1);
          if (e.key === "ArrowRight") setQuery(Number(qid.value) + 1);
        });

        setQuery(0);
      }

      main().catch(err => {
        console.error(err);
        document.body.innerHTML = "<pre>" + String(err) + "</pre>";
      });
    </script>
  </body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=Path, default=Path("./"))
    parser.add_argument("--model_dir", type=Path, default=Path("."))
    parser.add_argument("--images_dir", type=Path, default=Path("./test_images"))
    parser.add_argument("--out_dir", type=Path, default=Path("./results/encoder_query_attn"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_image_size", type=int, default=1024, choices=[768, 1024])
    parser.add_argument("--display_size", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=0, help="0 means process all images")
    parser.add_argument("--mass_threshold", type=float, default=0.6)
    parser.add_argument("--last_n_layers", type=int, default=2)
    parser.add_argument("--layers", type=str, default="all", choices=["last", "all"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_weight_file(args.model_dir, args.weights_dir)

    # Load full model once, but only use the vision encoder path.
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(args.device).to(torch.bfloat16)

    # Swap the qwen2 encoder to eager attention so attentions are available.
    sd = model.model.qwen2_model.state_dict()
    eager_qwen2 = deepencoderv2.build_qwen2_decoder_as_encoder(attn_implementation="eager")
    eager_qwen2.load_state_dict(sd, strict=True)
    eager_qwen2 = eager_qwen2.eval().to(args.device).to(torch.bfloat16)
    model.model.qwen2_model = eager_qwen2

    img_paths = _list_images(args.images_dir)
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    index_items = []
    for img_path in img_paths:
        stem = img_path.stem
        out_img_dir = args.out_dir / stem
        out_img_dir.mkdir(parents=True, exist_ok=True)

        fill = (127, 127, 127)
        display_img = _square_pad_contain(Image.open(img_path), args.display_size, fill)
        display_img_path = out_img_dir / "image.png"
        display_img.save(display_img_path)

        model_img = display_img.resize((args.model_image_size, args.model_image_size), resample=Image.BICUBIC)
        image_tensor = tfm(model_img).unsqueeze(0).to(args.device).to(torch.bfloat16)

        with torch.inference_mode():
            _, attentions = model.model.encode_global_view(image_tensor, return_attentions=True)

        if not attentions:
            raise RuntimeError(
                "Encoder attentions were not returned. "
                "Try setting attn_implementation='eager' for the qwen2 encoder."
            )

        layers_total = len(attentions)
        if args.layers == "all":
            layer_indices = list(range(layers_total))
        else:
            last_n = int(args.last_n_layers)
            start = max(0, layers_total - last_n)
            layer_indices = list(range(start, layers_total))

        seq_len = int(attentions[0].shape[-1])
        if seq_len % 2 != 0:
            raise ValueError(f"Unexpected seq_len for visual+query concat: {seq_len}")
        n_query = seq_len // 2

        layer_query_weights = []
        for li in layer_indices:
            attn_layer = attentions[li].mean(dim=1)[0]  # [S, S], mean over heads
            layer_query_weights.append(_extract_query_weights(attn_mean=attn_layer, n_query=n_query))

        data = {
            "image_name": img_path.name,
            "model_image_size": int(args.model_image_size),
            "display_size": int(args.display_size),
            "n_query": int(n_query),
            "grid_size": int(math.isqrt(n_query)),
            "layers_total": int(layers_total),
            "layer_indices": layer_indices,
            "layer_query_weights": layer_query_weights,
            "layers_used": int(len(layer_indices)),
            "mass_threshold": float(args.mass_threshold),
        }

        (out_img_dir / "data.js").write_text(
            "window.ENCODER_QUERY_ATTN_DATA = " + json.dumps(data, ensure_ascii=False) + ";",
            encoding="utf-8",
        )
        html = (
            HTML_TEMPLATE.replace("__LAST_N_LAYERS__", str(int(len(layer_indices))))
            .replace("__IMAGE_NAME__", img_path.name)
            .replace("__DISPLAY_SIZE__", str(int(args.display_size)))
        )
        (out_img_dir / "index.html").write_text(html, encoding="utf-8")

        index_items.append((img_path.name, f"{stem}/index.html"))

    index_html = ["<!doctype html><html><head><meta charset='utf-8'><title>Encoder Query Attention</title></head><body>"]
    index_html.append("<h2>Encoder Query Attention (global view only)</h2><ul>")
    for name, rel in index_items:
        index_html.append(f"<li><a href='{rel}'>{name}</a></li>")
    index_html.append("</ul></body></html>")
    (args.out_dir / "index.html").write_text("\n".join(index_html), encoding="utf-8")

    print(f"Saved HTML to: {args.out_dir}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
