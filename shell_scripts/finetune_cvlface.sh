#!/bin/bash
# Fine-tune CVLFace ViT-Base teacher on TinyFace
# Produces teacher baseline and fine-tuned checkpoint for KD
#
# Uses 112×112 input (teacher native resolution) with [-1,1] normalization.
# Low LR (1e-4) since backbone is already well-trained on WebFace4M.
# The evaluation callback reports rank@1 during training — this IS the teacher baseline.
#
# After training, note the checkpoint path (teacher_backbone.pt) for Part B runs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python training_pipeline_teacher_finetune.py \
    --teacher_type cvlface_vit_base \
    --img_size 112 --batch_size 8 --max_epochs 50 --seed 73 \
    --head_type adaface --head_scale 64 --head_margin 0.4 \
    --adaface_h 0.33 --adaface_t_alpha 0.01 \
    --lr 1e-4 --weight_decay 1e-5 --warmup_epochs 5 \
    --freeze_backbone_epochs 5 --backbone_lr_factor 0.1 \
    --rank_k 10 --wandb_project vit_book_teacher_finetune
