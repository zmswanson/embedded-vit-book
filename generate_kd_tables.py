"""Generate comparison tables for Phase 2.3 KD evaluation results."""
import csv

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

kd_rows = load_csv("inference_results/kd_cvlface_results.csv")
teacher_rows = load_csv("inference_results/teacher_baselines.csv")
phase1_rows = load_csv("inference_results/final_models_inference_results.csv")

kd = {r["wandb_id"]: float(r["tinyface/test/rank@1"]) for r in kd_rows}
teacher = {r["model_name"]: float(r["tinyface/test/rank@1"]) for r in teacher_rows}
p1 = {r["wandb_id"]: float(r["tinyface/test/rank@1"]) for r in phase1_rows}

# Phase 1 mapping (from shell_scripts/inference_final_models.sh)
p1_lora = {
    "swin_tiny": p1["f0frj93y"], "swin_small": p1["ymm0k0ub"],
    "swin_base": p1["ci53ufy8"], "deit3_small": p1["cocm8b7d"],
    "deit3_medium": p1["dpwn5kb1"], "deit3_base": p1["2ljjzfzb"],
}
p1_base = {
    "swin_tiny": p1["p09btsx3"], "swin_small": p1["rbeovm8i"],
    "swin_base": p1["xd7hukv3"], "deit3_small": p1["rdsdt7xb"],
    "deit3_medium": p1["k102lp3v"], "deit3_base": p1["vimriwcl"],
}
p1_best = {m: max(p1_base[m], p1_lora[m]) for m in p1_base}

models = ["swin_tiny", "swin_small", "swin_base", "deit3_small", "deit3_medium", "deit3_base"]
frozen_direct = {"swin_tiny": "y6ddjqyw", "swin_small": "o8hvv51y", "swin_base": "i0fg9wl7",
                 "deit3_small": "lkhiumos", "deit3_medium": "6tkurhll", "deit3_base": "f63auzxo"}
frozen_lora = {"swin_tiny": "j5t9bwdw", "swin_small": "oz4o3b3x", "swin_base": "fmfagyf4",
               "deit3_small": "wbo6pu5v", "deit3_medium": "kfbdjiaz", "deit3_base": "fz8pc2je"}
ft_direct = {"swin_tiny": "5wabjm9e", "swin_small": "sxo4xwy2", "swin_base": "jh5v6ltc",
             "deit3_small": "nz4gtmdt", "deit3_medium": "zhoquhp7", "deit3_base": "5mngq5jn"}
ft_lora = {"swin_tiny": "s2ceip2w", "swin_small": "ybx1z1y5", "swin_base": "i8c854l4",
           "deit3_small": "vsypj6cs", "deit3_medium": "0gt8dwde", "deit3_base": "3d7j4ck4"}

def fmt(v): return f"{v:.3f}"
def delta(v, ref):
    d = v - ref
    return f"{'+' if d >= 0 else ''}{d:.3f}"

lines = []
L = lines.append

L("# Phase 2.3 \u2014 CVLFace KD Evaluation Results")
L("")
L("TinyFace **test-set** rank@1 accuracy.  All students trained at 96\u00d796 with")
L("ImageNet-pretrained timm backbones.  Teacher is CVLFace ViT-Base (112\u00d7112).")
L("")

# Table 1
L("## Table 1 \u2014 Teacher Baselines")
L("")
L("| Model | rank@1 |")
L("|-------|--------|")
L(f"| CVLFace ViT-Base (frozen / off-the-shelf) | {fmt(teacher['cvlface_vit_base_frozen'])} |")
L(f"| CVLFace ViT-Base (fine-tuned on TinyFace) | {fmt(teacher['cvlface_vit_base_finetuned'])} |")
L("")

# Table 2
L("## Table 2 \u2014 KD with Frozen Teacher vs Phase 1")
L("")
L("| Student | Phase 1 Baseline | Phase 1 LoRA | Phase 1 Best | KD Direct (frozen) | KD LoRA (frozen) | KD Best (frozen) | \u0394 vs Phase 1 Best |")
L("|---------|-----------------|-------------|-------------|-------------------|-----------------|-----------------|------------------|")
for m in models:
    fd = kd[frozen_direct[m]]; fl = kd[frozen_lora[m]]; fb = max(fd, fl); pb = p1_best[m]
    L(f"| {m} | {fmt(p1_base[m])} | {fmt(p1_lora[m])} | {fmt(pb)} | {fmt(fd)} | {fmt(fl)} | {fmt(fb)} | {delta(fb, pb)} |")
L("")

# Table 3
L("## Table 3 \u2014 KD with Fine-tuned Teacher vs Phase 1")
L("")
L("| Student | Phase 1 Baseline | Phase 1 LoRA | Phase 1 Best | KD Direct (ft) | KD LoRA (ft) | KD Best (ft) | \u0394 vs Phase 1 Best |")
L("|---------|-----------------|-------------|-------------|---------------|-------------|-------------|------------------|")
for m in models:
    ftd = kd[ft_direct[m]]; ftl = kd[ft_lora[m]]; ftb = max(ftd, ftl); pb = p1_best[m]
    L(f"| {m} | {fmt(p1_base[m])} | {fmt(p1_lora[m])} | {fmt(pb)} | {fmt(ftd)} | {fmt(ftl)} | {fmt(ftb)} | {delta(ftb, pb)} |")
L("")

# Table 4
L("## Table 4 \u2014 Overall Comparison: Best KD vs Phase 1 Best")
L("")
L("| Student | Phase 1 Best | KD Best (frozen) | KD Best (ft) | Overall KD Best | Teacher | \u0394 vs Phase 1 |")
L("|---------|-------------|-----------------|-------------|----------------|---------|-------------|")
for m in models:
    fb = max(kd[frozen_direct[m]], kd[frozen_lora[m]])
    ftb = max(kd[ft_direct[m]], kd[ft_lora[m]])
    overall = max(fb, ftb)
    tl = "frozen" if fb >= ftb else "ft"
    pb = p1_best[m]
    L(f"| {m} | {fmt(pb)} | {fmt(fb)} | {fmt(ftb)} | **{fmt(overall)}** | {tl} | {delta(overall, pb)} |")
L("")

# Summary
L("## Summary")
L("")
L(f"- **Teacher frozen rank@1**: {fmt(teacher['cvlface_vit_base_frozen'])}")
L(f"- **Teacher fine-tuned rank@1**: {fmt(teacher['cvlface_vit_base_finetuned'])}")
L("")
beat = sum(1 for m in models if max(kd[frozen_direct[m]], kd[frozen_lora[m]], kd[ft_direct[m]], kd[ft_lora[m]]) > p1_best[m])
L(f"- **Students matching/exceeding Phase 1 best**: {beat}/6")
L("")
L("### Notes")
L("")
L("- Swin LoRA + KD runs show severe degradation (rank@1 ~0.35\u20130.42), consistent")
L("  with the Swin LoRA collapse observed in Phase 2.2 training.  These are included")
L("  for completeness but should not be interpreted as representative KD+LoRA results.")
L("- DeiT3 LoRA + KD runs performed normally (no collapse).")
L("- Direct (full fine-tune) KD generally outperformed LoRA KD.")
L("")

out = "\n".join(lines)
with open("inference_results/kd_cvlface_comparison.md", "w") as f:
    f.write(out)
print(out)
