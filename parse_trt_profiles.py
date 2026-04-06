#!/usr/bin/env python3
"""Phase 5.4 — Parse TensorRT profile JSONs into per-layer category breakdowns.

Usage:
    python parse_trt_profiles.py [--profiles-dir profiles/] [--out profiles/profiling_summary.csv]

Input:  profiles/*_profile.json  (from trtexec --exportProfile)
Output: profiles/profiling_summary.csv
        profiles/profiling_layers.csv  (raw per-layer rows for deep dives)
"""
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Layer categorisation
# TRT 10.3 fuses layers aggressively.  Mangled kernel names encode every fused
# op as an abbreviation:
#   Fc=FullyConnected, Mea=Mean, Sqr=Sqrt, Div=Div, Cas=Cast, Erf=Erf,
#   Tra=Transpose, Res=Reshape, Sli=Slice, Con=Concat, Rep=Repeat, Add/Sub/Mul
# Rules are ordered most-specific-first (first match wins).
# ---------------------------------------------------------------------------

CATEGORY_RULES = [
    # ---- Attention core ----
    # MHA scaled-dot-product kernel (v1=global/DeiT3, v2=windowed/Swin)
    ("attention", ["_gemm_mha_v1", "_gemm_mha_v2"]),
    # DeiT3 QKV reshape/split preceding MHA
    ("attention", ["TraSliSliSliResRes"]),
    # Swin window partition (position bias + cyclic shift)
    ("attention", ["AddSqrDivMulCasMulAddRepRepCon", "AddSqrDivMulCasMulAddSliSliCon"]),
    # Swin unpartition + residual + LN fused together
    ("attention", ["ResTraResTraResSliSli"]),
    # Swin fused 3-head QKV matmul (TRT name: __myeN+__myeM+__myeK)
    ("attention", ["+__mye"]),

    # ---- FFN (GELU activation is the definitive FFN-1 indicator) ----
    # FC followed by x*0.5*(1+erf(x/sqrt(2))) = GELU
    ("ffn", ["FcMulCasErfCasAddMulMul", "MulCasErfCasAddMulMul"]),

    # ---- Linear projections (attn output-proj + FFN-2 have identical structure) ----
    # Cannot distinguish without positional context in fused TRT graph.
    # DeiT3: node_MatMul = combined QKV;  Swin: node_MatMul = output-proj
    ("linear", ["node_MatMul", "FcMulAddCas", "FcAdd", "FcMulAdd"]),

    # ---- LayerNorm (all variants share: Mea->Sub->Mul->Mea->Add->Sqr->Div) ----
    ("norm", ["CasMeaSubMulMeaAddSqrDivMulCasMulAdd",
              "MulAddCasMeaSubMulMeaAddSqrDivMulCasMulAdd",
              "MulAddCasMeaSubMulMea",
              "CasMeaSubMulMea"]),

    # ---- Conv / patch embed / patch merging ----
    ("conv_embed", ["node_conv2d", "node_conv_", "node_linear_",
                    "/Conv", "downsample", "patch"]),

    # ---- TRT reformatting passes (near-zero cost) ----
    ("reformat", ["Reformatting", "CopyNode"]),
]

# Canonical display order for CSV columns
# Use dict.fromkeys to deduplicate while preserving insertion order
CATEGORIES = list(dict.fromkeys(c for c, _ in CATEGORY_RULES)) + ["other"]


def categorize_layer(name: str) -> str:
    # NOTE: match on original case (TRT names are mixed-case, keywords are exact)
    for cat, keywords in CATEGORY_RULES:
        if any(k in name for k in keywords):
            return cat
    return "other"


def parse_profile_json(path: Path) -> dict:
    """Return (cats_dict, total_ms, layer_rows) for one profile JSON."""
    data = json.loads(path.read_text())

    # TRT 10.x exportProfile JSON: list where first entry is {"count": N},
    # remaining entries are {"name":..., "averageMs":..., "medianMs":..., "percentage":...}
    cats = defaultdict(float)
    total = 0.0
    layer_rows = []

    for layer in data:
        name = layer.get("name", "")
        if not name:
            continue  # skip the leading {"count": N} metadata entry
        # TRT 10.x uses "averageMs"; older TRT uses "layerTime"
        t = float(layer.get("averageMs", layer.get("layerTime", 0)))
        cat = categorize_layer(name)
        cats[cat] += t
        total += t
        layer_rows.append({
            "name": name,
            "category": cat,
            "avg_ms": round(t, 4),
        })

    return cats, total, layer_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", default="profiles")
    parser.add_argument("--out", default="profiles/profiling_summary.csv")
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    json_files = sorted(profiles_dir.glob("*_profile.json"))

    if not json_files:
        print(f"No *_profile.json files found in {profiles_dir}/")
        return

    summaries = []
    all_layer_rows = []

    for f in json_files:
        model = f.stem.replace("_profile", "")
        cats, total, layer_rows = parse_profile_json(f)

        row = {"model": model, "total_ms": round(total, 3)}
        for cat in CATEGORIES:
            t = cats.get(cat, 0.0)
            row[f"{cat}_ms"] = round(t, 3)
            row[f"{cat}_pct"] = round(100 * t / total, 1) if total else 0.0
        summaries.append(row)

        for lr in layer_rows:
            lr["model"] = model
        all_layer_rows.extend(layer_rows)

        # Print human-readable summary line
        breakdown = "  ".join(
            f"{cat}={100*cats.get(cat,0)/total:.0f}%"
            for cat in CATEGORIES
            if cats.get(cat, 0) > 0
        )
        print(f"{model:<50} total={total:6.3f}ms  {breakdown}")

    # --- Write summary CSV ---
    summary_fields = ["model", "total_ms"]
    for cat in CATEGORIES:
        summary_fields += [f"{cat}_ms", f"{cat}_pct"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSummary CSV → {out_path}  ({len(summaries)} models)")

    # --- Write per-layer CSV ---
    layer_path = out_path.parent / "profiling_layers.csv"
    layer_fields = ["model", "name", "category", "avg_ms"]
    with open(layer_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=layer_fields)
        w.writeheader()
        w.writerows(all_layer_rows)
    print(f"Layers CSV  → {layer_path}  ({len(all_layer_rows)} layer rows)")


if __name__ == "__main__":
    main()
