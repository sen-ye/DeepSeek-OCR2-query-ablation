import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageAttnRecord:
    folder: str
    image_name: str
    n_query: int
    grid_size: int
    layer_indices: list[int]  # actual layer ids, length = L
    weights: np.ndarray  # [L, Q, Q], float32


def _parse_data_js(path: Path) -> dict[str, Any]:
    s = path.read_text(encoding="utf-8")
    m = re.search(r"ENCODER_QUERY_ATTN_DATA\s*=\s*(\{.*\})\s*;\s*$", s, flags=re.S)
    if not m:
        raise ValueError(f"Unrecognized data.js format: {path}")
    return json.loads(m.group(1))


def _load_record(folder: Path) -> ImageAttnRecord:
    data_path = folder / "data.js"
    data = _parse_data_js(data_path)

    required = {"image_name", "n_query", "grid_size", "layer_indices", "layer_query_weights"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"{data_path} missing keys: {sorted(missing)}")

    n_query = int(data["n_query"])
    grid_size = int(data["grid_size"])
    if grid_size * grid_size != n_query:
        raise ValueError(f"{data_path} has inconsistent grid_size/n_query: {grid_size}^2 != {n_query}")

    layer_indices = [int(x) for x in data["layer_indices"]]
    layer_query_weights = data["layer_query_weights"]
    if len(layer_query_weights) != len(layer_indices):
        raise ValueError(f"{data_path} has inconsistent layer lengths: weights vs indices")

    # Convert to compact float32 tensor: [L, Q, Q]
    weights = np.asarray(layer_query_weights, dtype=np.float32)
    if weights.ndim != 3 or weights.shape[1] != n_query or weights.shape[2] != n_query:
        raise ValueError(f"{data_path} unexpected weights shape: {weights.shape}")

    return ImageAttnRecord(
        folder=folder.name,
        image_name=str(data["image_name"]),
        n_query=n_query,
        grid_size=grid_size,
        layer_indices=layer_indices,
        weights=weights,
    )


def _find_subfolders(in_dir: Path) -> list[Path]:
    subfolders: list[Path] = []
    for p in sorted(in_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if (p / "data.js").exists():
            subfolders.append(p)
    return subfolders


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _cosine_pairwise(x: np.ndarray) -> np.ndarray:
    # x: [N, D] rows are vectors
    xx = x.astype(np.float64, copy=False)
    dot = xx @ xx.T
    norms = np.sqrt(np.clip(np.diag(dot), 0.0, None))
    denom = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dot / denom
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float64)


def _pearson_pairwise(x: np.ndarray) -> np.ndarray:
    # x: [N, D] rows are vectors; Pearson correlation across D.
    xx = x.astype(np.float64, copy=False)
    xx = xx - xx.mean(axis=1, keepdims=True)
    dot = xx @ xx.T
    norms = np.sqrt(np.clip(np.diag(dot), 0.0, None))
    denom = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dot / denom
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float64)


def _centroids_for_layer(weights_layer: np.ndarray, grid_size: int) -> np.ndarray:
    # weights_layer: [Q, Q] where each row is a distribution over Q visual tokens.
    # returns: [Q, 2] centroids in grid coordinates (cell centers): x,y in [0.5, g-0.5]
    g = grid_size
    coords_x = np.tile(np.arange(g, dtype=np.float32) + 0.5, g)  # [Q]
    coords_y = np.repeat(np.arange(g, dtype=np.float32) + 0.5, g)  # [Q]
    cx = weights_layer @ coords_x
    cy = weights_layer @ coords_y
    return np.stack([cx, cy], axis=1)  # [Q, 2]


def _centroid_l2_mean(ca: np.ndarray, cb: np.ndarray) -> float:
    # ca/cb: [Q, 2]
    d = ca.astype(np.float64) - cb.astype(np.float64)
    return float(np.sqrt((d * d).sum(axis=1)).mean())


def _centroid_l2_pairwise_per_query(centroids: np.ndarray) -> np.ndarray:
    # centroids: [N, Q, 2] -> [Q, N, N] L2 distance per query between images
    c = centroids.astype(np.float64, copy=False)
    diff = c[:, None, :, :] - c[None, :, :, :]  # [N,N,Q,2]
    d2 = (diff * diff).sum(axis=-1)  # [N,N,Q]
    d = np.sqrt(d2)  # [N,N,Q]
    return np.transpose(d, (2, 0, 1))  # [Q,N,N]


def _summ_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=Path, default=Path("./results/encoder_query_attn"))
    parser.add_argument("--out_dir", type=Path, default=Path("./results/analysis"))
    parser.add_argument("--max_images", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    _safe_mkdir(out_dir)

    subfolders = _find_subfolders(in_dir)
    if args.max_images and args.max_images > 0:
        subfolders = subfolders[: args.max_images]

    if not subfolders:
        raise SystemExit(f"No result subfolders found under: {in_dir}")

    records: list[ImageAttnRecord] = []
    skipped: list[tuple[str, str]] = []
    for folder in subfolders:
        try:
            records.append(_load_record(folder))
        except Exception as e:
            skipped.append((folder.name, f"{type(e).__name__}: {e}"))

    if skipped:
        (out_dir / "skipped.txt").write_text(
            "\n".join([f"{name}\t{reason}" for name, reason in skipped]) + "\n", encoding="utf-8"
        )

    if len(records) < 2:
        raise SystemExit(f"Need at least 2 valid subfolders to compare. Valid={len(records)}, skipped={len(skipped)}")

    # Group by n_query + grid_size for safety.
    groups: dict[tuple[int, int], list[ImageAttnRecord]] = {}
    for r in records:
        groups.setdefault((r.n_query, r.grid_size), []).append(r)

    meta = {
        "in_dir": str(in_dir),
        "n_total_subfolders": len(subfolders),
        "n_valid": len(records),
        "n_skipped": len(skipped),
        "groups": {f"nq{nq}_g{g}": [x.folder for x in rs] for (nq, g), rs in groups.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    group_index_items: list[tuple[str, str]] = []

    # Write outputs per group (usually only one).
    for (n_query, grid_size), group in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        if len(group) < 2:
            continue

        group_tag = f"nq{n_query}_g{grid_size}"
        group_dir = out_dir / group_tag
        _safe_mkdir(group_dir)
        group_index_items.append((group_tag, f"{group_tag}/index.html"))

        # Common layers (intersection by actual layer id)
        common_layers = set(group[0].layer_indices)
        for r in group[1:]:
            common_layers &= set(r.layer_indices)
        common_layers = set(sorted(common_layers))
        if not common_layers:
            raise SystemExit(f"Group {group_tag} has no common layers across images.")

        # Map actual layer id -> position in each record.
        layer_pos = [{li: i for i, li in enumerate(r.layer_indices)} for r in group]
        image_ids = [r.folder for r in group]
        image_names = [r.image_name for r in group]

        # Pairwise metrics per (layer, query) across images (records all results).
        pairwise_csv = group_dir / "pairwise_per_layer_query.csv"
        with pairwise_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["layer", "query", "img_a", "img_b", "cosine", "pearson", "centroid_l2_grid"])

            layer_summary_rows: list[dict[str, Any]] = []
            layer_query_summary_rows: list[dict[str, Any]] = []
            pairwise_overall_acc: dict[tuple[str, str], dict[str, list[float]]] = {}

            for li in sorted(common_layers):
                # Collect weights for this layer across images: W_layer is [N, Q, Q]
                W_list: list[np.ndarray] = []
                for rec, pos_map in zip(group, layer_pos):
                    p = pos_map[li]
                    W_list.append(rec.weights[p])
                W_layer = np.stack(W_list, axis=0).astype(np.float32, copy=False)

                # Centroids for all queries in this layer: [N, Q, 2]
                centroids_layer = np.stack(
                    [_centroids_for_layer(W_layer[i], grid_size) for i in range(W_layer.shape[0])], axis=0
                )
                centroid_l2_q = _centroid_l2_pairwise_per_query(centroids_layer)  # [Q,N,N]

                # Per-layer accumulators across queries and pairs
                layer_cos_vals: list[float] = []
                layer_pea_vals: list[float] = []
                layer_cen_vals: list[float] = []

                # Save per-layer data for HTML (upper triangle only, to keep size down)
                layer_data_dir = group_dir / "layer_data"
                _safe_mkdir(layer_data_dir)

                n = len(group)
                tri_i, tri_j = np.triu_indices(n, k=1)
                pairs = [[int(i), int(j)] for i, j in zip(tri_i.tolist(), tri_j.tolist())]
                p_count = len(pairs)
                cos_ut = np.empty((n_query, p_count), dtype=np.float32)
                pea_ut = np.empty((n_query, p_count), dtype=np.float32)
                cen_ut = np.empty((n_query, p_count), dtype=np.float32)

                for qid in range(n_query):
                    X = W_layer[:, qid, :]  # [N, Q]
                    cos_m = _cosine_pairwise(X)  # [N,N]
                    pea_m = _pearson_pairwise(X)  # [N,N]
                    cen_m = centroid_l2_q[qid]  # [N,N]

                    cos_row = cos_m[tri_i, tri_j]
                    pea_row = pea_m[tri_i, tri_j]
                    cen_row = cen_m[tri_i, tri_j]
                    cos_ut[qid] = cos_row.astype(np.float32, copy=False)
                    pea_ut[qid] = pea_row.astype(np.float32, copy=False)
                    cen_ut[qid] = cen_row.astype(np.float32, copy=False)

                    # Emit pair rows and collect stats for this (layer, query).
                    q_cos_vals = cos_row.astype(np.float64, copy=False).tolist()
                    q_pea_vals = pea_row.astype(np.float64, copy=False).tolist()
                    q_cen_vals = cen_row.astype(np.float64, copy=False).tolist()

                    for k, (i, j) in enumerate(zip(tri_i.tolist(), tri_j.tolist())):
                        c = float(cos_row[k])
                        p = float(pea_row[k])
                        d = float(cen_row[k])
                        w.writerow([li, qid, image_ids[i], image_ids[j], f"{c:.6f}", f"{p:.6f}", f"{d:.6f}"])

                        layer_cos_vals.append(c)
                        layer_pea_vals.append(p)
                        layer_cen_vals.append(d)

                        key = (image_ids[i], image_ids[j])
                        acc = pairwise_overall_acc.setdefault(key, {"cos": [], "pea": [], "cen": []})
                        acc["cos"].append(c)
                        acc["pea"].append(p)
                        acc["cen"].append(d)

                    # Per-(layer,query) summary over pairs.
                    csq = _summ_stats(q_cos_vals)
                    psq = _summ_stats(q_pea_vals)
                    dsq = _summ_stats(q_cen_vals)
                    layer_query_summary_rows.append(
                        {
                            "layer": li,
                            "query": qid,
                            "cosine_mean": csq["mean"],
                            "cosine_std": csq["std"],
                            "cosine_min": csq["min"],
                            "cosine_max": csq["max"],
                            "pearson_mean": psq["mean"],
                            "pearson_std": psq["std"],
                            "pearson_min": psq["min"],
                            "pearson_max": psq["max"],
                            "centroid_l2_mean": dsq["mean"],
                            "centroid_l2_std": dsq["std"],
                            "centroid_l2_min": dsq["min"],
                            "centroid_l2_max": dsq["max"],
                        }
                    )

                # Per-layer summary over all queries & pairs.
                cs = _summ_stats(layer_cos_vals)
                ps = _summ_stats(layer_pea_vals)
                ds = _summ_stats(layer_cen_vals)
                layer_summary_rows.append(
                    {
                        "layer": li,
                        "cosine_mean": cs["mean"],
                        "cosine_std": cs["std"],
                        "cosine_min": cs["min"],
                        "cosine_max": cs["max"],
                        "pearson_mean": ps["mean"],
                        "pearson_std": ps["std"],
                        "pearson_min": ps["min"],
                        "pearson_max": ps["max"],
                        "centroid_l2_mean": ds["mean"],
                        "centroid_l2_std": ds["std"],
                        "centroid_l2_min": ds["min"],
                        "centroid_l2_max": ds["max"],
                    }
                )

                # Write per-layer upper-triangle data for HTML
                layer_json_path = layer_data_dir / f"L{li}.json"
                layer_json_path.write_text(
                    json.dumps(
                        {
                            "layer": li,
                            "images": image_ids,
                            "n_query": n_query,
                            "pairs": pairs,
                            "cosine_ut": np.round(cos_ut.astype(np.float64), 6).tolist(),
                            "pearson_ut": np.round(pea_ut.astype(np.float64), 6).tolist(),
                            "centroid_l2_ut": np.round(cen_ut.astype(np.float64), 6).tolist(),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

        # Layer summary CSV
        layer_summary_csv = group_dir / "layer_summary.csv"
        with layer_summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "layer",
                    "cosine_mean",
                    "cosine_std",
                    "cosine_min",
                    "cosine_max",
                    "pearson_mean",
                    "pearson_std",
                    "pearson_min",
                    "pearson_max",
                    "centroid_l2_mean",
                    "centroid_l2_std",
                    "centroid_l2_min",
                    "centroid_l2_max",
                ]
            )
            for row in sorted(layer_summary_rows, key=lambda x: x["layer"]):
                w.writerow(
                    [
                        row["layer"],
                        f"{row['cosine_mean']:.6f}",
                        f"{row['cosine_std']:.6f}",
                        f"{row['cosine_min']:.6f}",
                        f"{row['cosine_max']:.6f}",
                        f"{row['pearson_mean']:.6f}",
                        f"{row['pearson_std']:.6f}",
                        f"{row['pearson_min']:.6f}",
                        f"{row['pearson_max']:.6f}",
                        f"{row['centroid_l2_mean']:.6f}",
                        f"{row['centroid_l2_std']:.6f}",
                        f"{row['centroid_l2_min']:.6f}",
                        f"{row['centroid_l2_max']:.6f}",
                    ]
                )

        # Per-(layer,query) summary CSV (stats over pairs).
        layer_query_summary_csv = group_dir / "layer_query_summary.csv"
        with layer_query_summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "layer",
                    "query",
                    "cosine_mean",
                    "cosine_std",
                    "cosine_min",
                    "cosine_max",
                    "pearson_mean",
                    "pearson_std",
                    "pearson_min",
                    "pearson_max",
                    "centroid_l2_mean",
                    "centroid_l2_std",
                    "centroid_l2_min",
                    "centroid_l2_max",
                ]
            )
            for row in sorted(layer_query_summary_rows, key=lambda x: (x["layer"], x["query"])):
                w.writerow(
                    [
                        row["layer"],
                        row["query"],
                        f"{row['cosine_mean']:.6f}",
                        f"{row['cosine_std']:.6f}",
                        f"{row['cosine_min']:.6f}",
                        f"{row['cosine_max']:.6f}",
                        f"{row['pearson_mean']:.6f}",
                        f"{row['pearson_std']:.6f}",
                        f"{row['pearson_min']:.6f}",
                        f"{row['pearson_max']:.6f}",
                        f"{row['centroid_l2_mean']:.6f}",
                        f"{row['centroid_l2_std']:.6f}",
                        f"{row['centroid_l2_min']:.6f}",
                        f"{row['centroid_l2_max']:.6f}",
                    ]
                )

        # Pairwise overall (mean across all common layers and all queries).
        overall_csv = group_dir / "pairwise_overall.csv"
        overall_rows: list[dict[str, Any]] = []
        for (a, b), acc in pairwise_overall_acc.items():
            overall_rows.append(
                {
                    "img_a": a,
                    "img_b": b,
                    "cosine_mean": float(np.mean(acc["cos"])),
                    "pearson_mean": float(np.mean(acc["pea"])),
                    "centroid_l2_mean": float(np.mean(acc["cen"])),
                }
        )
        overall_rows.sort(key=lambda x: (-x["cosine_mean"], -x["pearson_mean"], x["centroid_l2_mean"]))
        with overall_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "img_a",
                    "img_b",
                    "cosine_mean_over_layers_queries",
                    "pearson_mean_over_layers_queries",
                    "centroid_l2_mean_over_layers_queries",
                ]
            )
            for r in overall_rows:
                w.writerow(
                    [
                        r["img_a"],
                        r["img_b"],
                        f"{r['cosine_mean']:.6f}",
                        f"{r['pearson_mean']:.6f}",
                        f"{r['centroid_l2_mean']:.6f}",
                    ]
                )

        # Per-image average similarity (mean over pairs; pair metrics are already mean over layers&queries).
        per_image: dict[str, dict[str, list[float]]] = {img: {"cos": [], "pea": [], "cen": []} for img in image_ids}
        for r in overall_rows:
            a, b = r["img_a"], r["img_b"]
            per_image[a]["cos"].append(r["cosine_mean"])
            per_image[b]["cos"].append(r["cosine_mean"])
            per_image[a]["pea"].append(r["pearson_mean"])
            per_image[b]["pea"].append(r["pearson_mean"])
            per_image[a]["cen"].append(r["centroid_l2_mean"])
            per_image[b]["cen"].append(r["centroid_l2_mean"])

        per_image_rows: list[dict[str, Any]] = []
        for img in image_ids:
            per_image_rows.append(
                {
                    "img": img,
                    "image_name": image_names[image_ids.index(img)],
                    "cosine_mean": float(np.mean(per_image[img]["cos"])) if per_image[img]["cos"] else float("nan"),
                    "pearson_mean": float(np.mean(per_image[img]["pea"])) if per_image[img]["pea"] else float("nan"),
                    "centroid_l2_mean": float(np.mean(per_image[img]["cen"])) if per_image[img]["cen"] else float("nan"),
                }
            )
        per_image_rows.sort(key=lambda x: (-x["cosine_mean"], -x["pearson_mean"], x["centroid_l2_mean"]))
        per_image_csv = group_dir / "per_image_summary.csv"
        with per_image_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "img",
                    "image_name",
                    "cosine_mean_over_layers_queries",
                    "pearson_mean_over_layers_queries",
                    "centroid_l2_mean_over_layers_queries",
                ]
            )
            for r in per_image_rows:
                w.writerow(
                    [
                        r["img"],
                        r["image_name"],
                        f"{r['cosine_mean']:.6f}",
                        f"{r['pearson_mean']:.6f}",
                        f"{r['centroid_l2_mean']:.6f}",
                    ]
                )

        # HTML visualization for per-(layer,query) matrices
        layers_sorted = sorted(common_layers)
        html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Attention Similarity ({group_tag})</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
      .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
      select, input[type="range"] {{ width: 320px; }}
      table {{ border-collapse: collapse; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
      th.sticky {{ position: sticky; top: 0; background: #fafafa; z-index: 1; }}
      td.label {{ text-align: left; background: #fafafa; }}
      .small {{ font-size: 12px; color: #444; }}
      .grid {{ overflow: auto; max-width: 100%; }}
      .legend {{ display: flex; align-items: center; gap: 10px; }}
      .bar {{ width: 180px; height: 10px; background: linear-gradient(90deg, #2b83ba, #f7f7f7, #d7191c); border: 1px solid #ddd; }}
      .bar2 {{ width: 180px; height: 10px; background: linear-gradient(90deg, #d7191c, #f7f7f7); border: 1px solid #ddd; }}
    </style>
  </head>
  <body>
    <h2>Attention Similarity ({group_tag})</h2>
    <div class="small mono">
      Images: {len(image_ids)} | n_query: {n_query} | grid: {grid_size}x{grid_size} | layers: {", ".join(str(x) for x in layers_sorted)}
    </div>
    <div style="height: 12px;"></div>
    <div class="row">
      <div>
        <div><b>Metric</b></div>
        <select id="metric">
          <option value="cosine">cosine</option>
          <option value="pearson">pearson</option>
          <option value="centroid_l2">centroid_l2</option>
        </select>
      </div>
      <div>
        <div><b>Layer</b></div>
        <select id="layer"></select>
      </div>
      <div>
        <div><b>Query</b>: <span class="mono" id="qid_label"></span></div>
        <input id="qid" type="range" min="0" max="{n_query - 1}" step="1" value="0" />
        <div style="height: 6px;"></div>
        <select id="qid_select"></select>
      </div>
      <div class="legend">
        <div id="legend_div" class="bar"></div>
        <div class="small" id="legend_txt"></div>
      </div>
    </div>
    <div style="height: 12px;"></div>
    <div class="grid" id="matrix"></div>
    <div style="height: 12px;"></div>
    <div>
      <div><b>Pairs</b> (upper triangle, sorted by metric):</div>
      <pre class="mono" id="pairs"></pre>
    </div>
    <script>
      const layers = {json.dumps(layers_sorted)};
      const images = {json.dumps(image_ids)};
      const layerSel = document.getElementById("layer");
      const metricSel = document.getElementById("metric");
      const qid = document.getElementById("qid");
      const qidSelect = document.getElementById("qid_select");
      const qidLabel = document.getElementById("qid_label");
      const matrixDiv = document.getElementById("matrix");
      const pairsPre = document.getElementById("pairs");
      const legendDiv = document.getElementById("legend_div");
      const legendTxt = document.getElementById("legend_txt");

      for (const li of layers) {{
        const opt = document.createElement("option");
        opt.value = String(li);
        opt.textContent = "L" + String(li);
        layerSel.appendChild(opt);
      }}
      for (let i = 0; i < Number(qid.max) + 1; i++) {{
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = String(i);
        qidSelect.appendChild(opt);
      }}

      function clamp01(x) {{
        return Math.max(0.0, Math.min(1.0, x));
      }}

      function colorFor(metric, v) {{
        if (metric === "centroid_l2") {{
          // smaller is better -> red smaller, white larger
          const t = 1.0 - clamp01(v / ({grid_size} * 2.0));
          const r = Math.round(247 + (215 - 247) * t);
          const g = Math.round(247 + (25 - 247) * t);
          const b = Math.round(247 + (28 - 247) * t);
          return `rgb(${{r}},${{g}},${{b}})`;
        }}
        // cosine in [0..1], pearson in [-1..1]
        let t = 0.5;
        if (metric === "cosine") {{
          t = clamp01(v);
        }} else {{
          t = clamp01((v + 1.0) * 0.5);
        }}
        // blue-white-red
        const r = Math.round(43 + (215 - 43) * t);
        const g = Math.round(131 + (25 - 131) * t);
        const b = Math.round(186 + (28 - 186) * t);
        return `rgb(${{r}},${{g}},${{b}})`;
      }}

      function fmt(metric, v) {{
        if (!isFinite(v)) return "nan";
        return (metric === "centroid_l2") ? v.toFixed(3) : v.toFixed(4);
      }}

      function legend(metric) {{
        if (metric === "centroid_l2") {{
          legendDiv.className = "bar2";
          legendTxt.textContent = "centroid_l2: smaller is better (red=smaller)";
        }} else {{
          legendDiv.className = "bar";
          legendTxt.textContent = (metric === "cosine") ? "cosine: larger is better (red=larger)" : "pearson: larger is better (red=larger)";
        }}
      }}

      async function loadLayer(li) {{
        const url = `layer_data/L${{li}}.json`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error("Failed to fetch " + url + ": " + resp.status);
        return await resp.json();
      }}

      function buildMatrixFromUT(layerData, metric, q) {{
        const n = images.length;
        const m = Array.from({{length: n}}, (_, i) => Array.from({{length: n}}, (_, j) => (i === j ? 1.0 : 0.0)));
        const pairs = layerData.pairs;
        const ut = (metric === "cosine") ? layerData.cosine_ut[q]
                 : (metric === "pearson") ? layerData.pearson_ut[q]
                 : layerData.centroid_l2_ut[q];
        for (let k = 0; k < pairs.length; k++) {{
          const i = pairs[k][0];
          const j = pairs[k][1];
          const v = ut[k];
          m[i][j] = v;
          m[j][i] = v;
        }}
        if (metric === "centroid_l2") {{
          for (let i = 0; i < n; i++) m[i][i] = 0.0;
        }}
        return m;
      }}

      function renderMatrix(metric, m) {{
        const n = images.length;
        let html = "<table><tr><th class='sticky'></th>";
        for (let j = 0; j < n; j++) {{
          html += `<th class='sticky mono'>${{images[j]}}</th>`;
        }}
        html += "</tr>";
        for (let i = 0; i < n; i++) {{
          html += `<tr><td class='label mono'>${{images[i]}}</td>`;
          for (let j = 0; j < n; j++) {{
            const v = m[i][j];
            const bg = colorFor(metric, v);
            html += `<td style="background:${{bg}}" class="mono">${{fmt(metric, v)}}</td>`;
          }}
          html += "</tr>";
        }}
        html += "</table>";
        matrixDiv.innerHTML = html;
      }}

      function renderPairs(metric, m) {{
        const n = images.length;
        const pairs = [];
        for (let i = 0; i < n; i++) {{
          for (let j = i + 1; j < n; j++) {{
            pairs.push({{a: images[i], b: images[j], v: m[i][j]}});
          }}
        }}
        pairs.sort((x, y) => {{
          if (metric === "centroid_l2") return x.v - y.v;
          return y.v - x.v;
        }});
        pairsPre.textContent = pairs.map(p => `${{p.a}} vs ${{p.b}}: ${{fmt(metric, p.v)}}`).join("\\n");
      }}

      async function refresh() {{
        const li = Number(layerSel.value);
        const q = Number(qid.value);
        const metric = metricSel.value;
        qidSelect.value = String(q);
        qidLabel.textContent = String(q);
        legend(metric);
        if (!window.__LAYER_CACHE) window.__LAYER_CACHE = {{}};
        if (!window.__LAYER_CACHE[li]) {{
          window.__LAYER_CACHE[li] = await loadLayer(li);
        }}
        const layerData = window.__LAYER_CACHE[li];
        const m = buildMatrixFromUT(layerData, metric, q);
        renderMatrix(metric, m);
        renderPairs(metric, m);
      }}

      qid.addEventListener("input", () => refresh().catch(console.error));
      qidSelect.addEventListener("change", () => {{
        qid.value = qidSelect.value;
        refresh().catch(console.error);
      }});
      layerSel.addEventListener("change", () => refresh().catch(console.error));
      metricSel.addEventListener("change", () => refresh().catch(console.error));

      layerSel.value = String(layers[0]);
      qid.value = "0";
      refresh().catch(err => {{
        console.error(err);
        document.body.innerHTML = "<pre class='mono'>" + String(err) + "</pre>";
      }});
    </script>
  </body>
</html>
"""
        (group_dir / "index.html").write_text(html, encoding="utf-8")

        # Markdown summary
        def _fmt_pair(r: dict[str, Any]) -> str:
            return (
                f"{r['img_a']} vs {r['img_b']}: "
                f"cos={r['cosine_mean']:.4f}, pearson={r['pearson_mean']:.4f}, "
                f"centroidL2={r['centroid_l2_mean']:.4f}"
            )

        best_cos = max(overall_rows, key=lambda x: x["cosine_mean"])
        worst_cos = min(overall_rows, key=lambda x: x["cosine_mean"])
        best_pea = max(overall_rows, key=lambda x: x["pearson_mean"])
        worst_pea = min(overall_rows, key=lambda x: x["pearson_mean"])
        best_cen = min(overall_rows, key=lambda x: x["centroid_l2_mean"])
        worst_cen = max(overall_rows, key=lambda x: x["centroid_l2_mean"])

        md_lines = []
        md_lines.append(f"# Attention Similarity Analysis ({group_tag})")
        md_lines.append("")
        md_lines.append("## Inputs")
        md_lines.append(f"- in_dir: `{in_dir}`")
        md_lines.append(f"- images: {len(group)}")
        md_lines.append(f"- n_query: {n_query} (grid {grid_size}x{grid_size})")
        md_lines.append(f"- common_layers: {len(common_layers)} -> {', '.join(str(x) for x in sorted(common_layers))}")
        md_lines.append("")
        md_lines.append("## Overall (mean over common layers and all queries)")
        md_lines.append(f"- Most similar by cosine: {_fmt_pair(best_cos)}")
        md_lines.append(f"- Least similar by cosine: {_fmt_pair(worst_cos)}")
        md_lines.append(f"- Most similar by pearson: {_fmt_pair(best_pea)}")
        md_lines.append(f"- Least similar by pearson: {_fmt_pair(worst_pea)}")
        md_lines.append(f"- Most similar by centroid L2 (smaller is better): {_fmt_pair(best_cen)}")
        md_lines.append(f"- Least similar by centroid L2 (larger is worse): {_fmt_pair(worst_cen)}")
        md_lines.append("")
        md_lines.append("## Layer-wise summary")
        md_lines.append("See `layer_summary.csv` (stats over queries & pairs) and `layer_query_summary.csv` (stats over pairs per query).")
        md_lines.append("")
        md_lines.append("## Per-image average similarity (mean over pairs; pairs already mean over layers&queries)")
        md_lines.append("See `per_image_summary.csv`.")
        md_lines.append("")
        md_lines.append("## Visualization")
        md_lines.append("- Open `index.html` to browse per-(layer,query) pairwise matrices.")
        md_lines.append("")
        md_lines.append("## Notes")
        md_lines.append("- cosine/pearson are computed on the [Q] attention weights vector for the same layer and same query.")
        md_lines.append("- centroid uses grid coordinates (cell centers): x,y in [0.5, g-0.5].")
        md_lines.append("- centroid L2 is per-query; overall averages over all common layers and all queries.")
        md_lines.append("")

        (group_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if group_index_items:
        idx_lines = [
            "<!doctype html><html><head><meta charset='utf-8'><title>Analysis Index</title></head><body>",
            "<h2>Attention Similarity Analysis</h2><ul>",
        ]
        for name, rel in group_index_items:
            idx_lines.append(f"<li><a href='{rel}'>{name}</a></li>")
        idx_lines.append("</ul></body></html>")
        (out_dir / "index.html").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    print(f"Wrote analysis to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
