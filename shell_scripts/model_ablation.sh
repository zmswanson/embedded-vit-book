#!/bin/bash

model_thaw_list=(
    "--model convnext_base.fb_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model convnext_small.fb_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model convnext_tiny.fb_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model deit3_base_patch16_224.fb_in1k --gradual_unfreeze block --thaw_N_blocks 8"
    "--model deit3_medium_patch16_224.fb_in1k --gradual_unfreeze block --thaw_N_blocks 8"
    "--model deit3_small_patch16_224.fb_in1k --gradual_unfreeze block --thaw_N_blocks 8"
    "--model pvt_v2_b5.in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model pvt_v2_b3.in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model pvt_v2_b2.in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model resnet200d.ra2_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model resnet101d.ra2_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model resnet50d.ra2_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model swin_base_patch4_window7_224.ms_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model swin_small_patch4_window7_224.ms_in1k --gradual_unfreeze stage --thaw_N_stages 2"
    "--model swin_tiny_patch4_window7_224.ms_in1k--gradual_unfreeze stage --thaw_N_stages 2"
    "--model mobilevitv2_200.cvnets_in1k --gradual_unfreeze stage --thaw_N_stages 3"
)

# make sure to run from the root directory if dirname has "shell_scripts"
if [[ "$(dirname "$0")" == *"shell_scripts"* ]]; then
    cd "$(dirname "$0")/.."
fi

for model_thaw in "${model_thaw_list[@]}"; do
    echo "Running model ablation with settings: $model_thaw"
    python training_pipeline.py ${model_thaw} --img_size 96 --batch_size 8 \
        --max_epochs 50 --lr 5.782e-4 --weight_decay 1.0e-5 \
        --backbone_dropout 0.165 --head_type adaface --head_scale 96 \
        --head_margin 0.357 --adaface_h 0.2015 --adaface_t_alpha 0.0438 \
        --warmup_epochs 8 --unfreeze_every_epochs 5  --rank_k 10 \
        --wandb_project model_ablation_study --no_wandb_model
done