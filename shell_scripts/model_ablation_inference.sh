#!/bin/bash

model_list=(
    "--wandb_id kkj2mwz2 --model_name deit3_medium_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/kkj2mwz2/epoch=44-rank1tinyface/eval/rank@1=0.585.ckpt"
    "--wandb_id 0yo0e1l7 --model_name swin_small_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/0yo0e1l7/epoch=41-rank1tinyface/eval/rank@1=0.581.ckpt"
    "--wandb_id w2k02mix --model_name deit3_base_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/w2k02mix/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt"
    "--wandb_id 3faz5a7o --model_name swin_base_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/3faz5a7o/epoch=38-rank1tinyface/eval/rank@1=0.570.ckpt"
    "--wandb_id tjip2s4w --model_name deit3_small_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/tjip2s4w/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt"
    "--wandb_id nvzory9h --model_name pvt_v2_b5.in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/nvzory9h/epoch=21-rank1tinyface/eval/rank@1=0.566.ckpt"
    "--wandb_id y11v279f --model_name swin_tiny_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/y11v279f/epoch=32-rank1tinyface/eval/rank@1=0.572.ckpt"
    "--wandb_id hzvjhovx --model_name convnext_tiny.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/hzvjhovx/epoch=34-rank1tinyface/eval/rank@1=0.568.ckpt"
    "--wandb_id eltlatns --model_name pvt_v2_b3.in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/eltlatns/epoch=21-rank1tinyface/eval/rank@1=0.566.ckpt"
    "--wandb_id ccc9z3g5 --model_name convnext_small.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/ccc9z3g5/epoch=34-rank1tinyface/eval/rank@1=0.556.ckpt"
    "--wandb_id uvu73fxo --model_name pvt_v2_b2.in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/uvu73fxo/epoch=24-rank1tinyface/eval/rank@1=0.548.ckpt"
    "--wandb_id m4xjjh07 --model_name convnext_base.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/m4xjjh07/epoch=17-rank1tinyface/eval/rank@1=0.570.ckpt"
    "--wandb_id amsv29wr --model_name mobilevitv2_200.cvnets_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/amsv29wr/epoch=42-rank1tinyface/eval/rank@1=0.525.ckpt"
    "--wandb_id 4860x6ia --model_name resnet50d.ra2_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/4860x6ia/epoch=29-rank1tinyface/eval/rank@1=0.502.ckpt"
    "--wandb_id qopz1ks6 --model_name resnet101d.ra2_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/qopz1ks6/epoch=44-rank1tinyface/eval/rank@1=0.512.ckpt"
    "--wandb_id asyv1awr --model_name resnet200d.ra2_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/asyv1awr/epoch=33-rank1tinyface/eval/rank@1=0.484.ckpt"
    "--wandb_id 4860x6ia --model_name resnet50d.ra2_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/4860x6ia/epoch=29-rank1tinyface/eval/rank@1=0.502.ckpt"
    "--wandb_id qopz1ks6 --model_name resnet101d.ra2_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/qopz1ks6/epoch=44-rank1tinyface/eval/rank@1=0.512.ckpt"
    "--wandb_id asyv1awr --model_name resnet200d.ra2_in1k --ckpt_path /mnt/data/wandb/checkpoints/model_ablation_study/asyv1awr/epoch=33-rank1tinyface/eval/rank@1=0.484.ckpt"
)

save_path="./inference_results/model_ablation_inference_results.csv"

# make sure to run from the root directory if dirname has "shell_scripts"
if [[ "$(dirname "$0")" == *"shell_scripts"* ]]; then
    cd "$(dirname "$0")/.."
fi

for model_args in "${model_list[@]}"; do
    echo "Running inference with settings: $model_args"
    python inference.py ${model_args} --img_size 96 --batch_size 64 \
        --save_path "${save_path}"
done