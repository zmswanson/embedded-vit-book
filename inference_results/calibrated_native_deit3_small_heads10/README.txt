Calibrated native TensorRT INT8 comparison, 2026-10-22

Model: DeiT3-S baseline with 10% head pruning, batch 1, 96x96.
Results: comparison.csv and comparison.json.

The initial calibrated build allowed INT8 and FP16. Its detailed inspector
shows floating-point execution throughout. It is retained as a diagnostic;
opportunistic_timing.json must not be presented as an actual INT8 result.
opportunistic_build_script.py is the exact source used for that build.

required_int8/ is the verified integer-projection build. Its inspector records
48 learned linear projections using INT8 activations, INT8 weights, and
integer hardware tactics. Other layers use floating point. The ONNX layer
selection is recorded in int8_layer_names.json. This engine uses the same
real entropy-calibration scales as the initial build, verified by hashes.

Calibration uses 512 training identities, one image each, seed 73, excluding
validation split 9 and all test identities. calibration_manifest.json hashes
each source image, preprocessed tensor, split definition, and preparation
script. The 54-MiB tensor file can be regenerated from the local TinyFace
dataset with scripts/prepare_native_calibration.py. It is kept in
/tmp/scalable-vits-calibration on the workstation and Jetson.

Native accuracy uses exactly the same 4,443 Gallery_Match images and 3,728
probes for both engines. No distractors are included: these are results under
the existing chapter's reduced-gallery protocol. The input archive is
/tmp/scalable-vits-native-evaluation.zip on the Jetson; its hash and
preprocessing are recorded in the accuracy report.

The usable INT8 engine remains on jetson-orin at:
/ssd/vit_benchmark/results/deit3_small_heads10_required_int8/deit3_small_heads10_calibrated_int8.engine

Run the saved-result audit from the repository root:
  python3 scripts/audit_native_int8.py
  python3 scripts/audit_chapter_numerics.py

The first verifies calibration provenance, actual INT8 kernel formats,
engine hashes, clock settings, paired predictions and means; it regenerates
comparison.json and comparison.csv. The second incorporates the comparison
and evidence hashes into the chapter's numeric verification.

Builder invocation on the Jetson (use a fresh output directory to rebuild):
  python3 /tmp/build_calibrated_native_int8.py \
    --onnx /ssd/vit_benchmark/onnx_models/pruned_v2/deit3_small_baseline-heads_0.10.onnx \
    --calibration-dir /tmp/scalable-vits-calibration \
    --output-dir /ssd/vit_benchmark/results/deit3_small_heads10_required_int8 \
    --require-int8-linear --int8-layer-names /tmp/int8_layer_names.json \
    --reuse-calibration-build /ssd/vit_benchmark/results/deit3_small_heads10_calibrated_int8

No training, package changes, or test-image calibration was performed.
