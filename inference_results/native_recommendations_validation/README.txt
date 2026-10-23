Native validation of the five Table 11 deployment engines

native_recommendations_accuracy.json contains direct TensorRT inference on
4,443 matching-gallery images and 3,728 probes, using the same decoded RGB
pixels, PIL bilinear resizing to 96x96, and ImageNet normalization as the
earlier native DeiT3-S evaluation. Distractors are not included. This closes
the deployment-backend accuracy gap. The full TinyFace distractor benchmark
is not rerun; the manuscript now explicitly labels the reduced gallery.
Each engine and the input archive are hashed; per-probe predictions are saved.

The scripts use CUDA runtime calls and TensorRT 10.3, with no new packages:
  scripts/evaluate_native_recommendations.py
  scripts/evaluate_native_deployment.py (unchanged inference/preprocessing helper)

The three previously unverified native scores are:
  Swin-T KD: 2,296/3,728 = 0.6158798283261803.
  Swin-T KD, two blocks dropped: 2,173/3,728 = 0.5828862660944206.
  ResNet-50d: 1,775/3,728 = 0.4761266094420601.
The DeiT3-S FP16/INT8 scores reproduce their earlier native evaluations.

timing/ holds a refreshed three-repeat trtexec benchmark of the same engines,
using the chapter's native GPU-compute scope, with no transfers and CUDA
Graph. The detailed energy/memory traces are stored in the sibling directory
native_recommendations_resources/. Measurements are combined and verified by
scripts/audit_native_recommendations.py into deployment_measurements.json/.csv.
The chapter's general numeric audit consumes that verified summary.

Timing logs and other raw evidence are local artifacts excluded from Git.
The full audit requires those files and verifies decompressed .log.gz bytes
when archives are available; uncompressed .log files are optional.

The fresh timing means are 1.34477, 1.16831, 0.94996, 1.02404 and 1.49715 ms
in table order. Swin-T and ResNet execute faster than in the archived April
power-mode sweep; the reason is not isolated. Both sets retain their own raw
logs and scope. CPU/GPU limits are verified, but current EMC clock readback
is unavailable to this login; no memory-clock explanation is claimed. Table
11 uses the entire fresh paired set, rather than combining the old and new
sessions. Table 10 and its historical cross-mode comparisons are preserved.
