# LoRA Sweep Analysis Summary

---

## swin_lora_adaface

- **File**: `wandb_report-runs_sweep-vit_book_swin_lora_ada.csv`
- **Model**: `swin_base_patch4_window7_224.ms_in1k`
- **Head**: adaface
- **Total runs**: 100 (including header row)
- **Successful runs**: 100
- **Failed/NaN runs**: 0 (0.0%)

### Swept parameters
`lr`, `head_scale`, `head_margin`, `adaface_h`, `adaface_t_alpha`

### Fixed LoRA parameters
- `lora_r`: 24
- `lora_alpha`: 120
- `lora_dropout`: 0.05
- `lora_qkv_proj`: 1
- `lora_train_bias`: 0
- `weight_decay`: 0.0001

### Top 5 Runs

| Rank | Name | rank@1 | rank@5 | lr | head_scale | head_margin | adaface_h | adaface_t_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00019313294723680936-adaface:s96.0:m0.30762863249271644:h0.3206180757713378:ta0.023366334755827948-lora(r24-a120.0-d0.05),qkv-thawed() | 0.5814 | 0.7171 | 1.93e-04 | 96 | 0.3076 | 0.3206 | 0.02337 |
| 2 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00032733700601098584-adaface:s96.0:m0.294341252075183:h0.23780790211193145:ta0.0116144441279277-lora(r24-a120.0-d0.05),qkv-thawed() | 0.5736 | 0.7074 | 3.27e-04 | 96 | 0.2943 | 0.2378 | 0.01161 |
| 3 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00026521667544675075-adaface:s96.0:m0.31060051178621:h0.30218766345333636:ta0.016059084816046794-lora(r24-a120.0-d0.05),qkv-thawed() | 0.5736 | 0.7326 | 2.65e-04 | 96 | 0.3106 | 0.3022 | 0.01606 |
| 4 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.000240110639492224-adaface:s96.0:m0.3372882450103749:h0.4018653164948164:ta0.012452258389872472-lora(r24-a120.0-d0.05),qkv-thawed() | 0.5717 | 0.7093 | 2.40e-04 | 96 | 0.3373 | 0.4019 | 0.01245 |
| 5 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00017042799065937635-adaface:s96.0:m0.29089348703703516:h0.3646457986566556:ta0.017188169320259728-lora(r24-a120.0-d0.05),qkv-thawed() | 0.5717 | 0.7074 | 1.70e-04 | 96 | 0.2909 | 0.3646 | 0.01719 |

### Hyperparameter Clustering (top 5)
- `lr`: min=0.0001704, max=0.0003273, std=6.184e-05
- `head_scale`: min=96, max=96, std=0
- `head_margin`: min=0.2909, max=0.3373, std=0.01833
- `adaface_h`: min=0.2378, max=0.4019, std=0.06249
- `adaface_t_alpha`: min=0.01161, max=0.02337, std=0.004675

### Full Distribution Stats
- `lr`: mean=0.0001446, std=0.0001177, range=[1.084e-05, 0.0004922]
- `head_scale`: mean=73.92, std=22.26, range=[32, 96]
- `head_margin`: mean=0.2915, std=0.05207, range=[0.2, 0.3961]
- `adaface_h`: mean=0.3439, std=0.07001, range=[0.2075, 0.4498]
- `adaface_t_alpha`: mean=0.01846, std=0.01195, range=[0.005035, 0.04874]

### Selected Best Config
**Run**: `swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00019313294723680936-adaface:s96.0:m0.30762863249271644:h0.3206180757713378:ta0.023366334755827948-lora(r24-a120.0-d0.05),qkv-thawed()` — **rank@1 = 0.5814**

---

## swin_lora_arcface

- **File**: `wandb_report-runs_sweep-vit_book_swin_lora.csv`
- **Model**: `swin_base_patch4_window7_224.ms_in1k`
- **Head**: arcface
- **Total runs**: 100 (including header row)
- **Successful runs**: 100
- **Failed/NaN runs**: 0 (0.0%)

### Swept parameters
`lr`, `weight_decay`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_qkv_proj`, `lora_train_bias`, `head_scale`

### Top 5 Runs

| Rank | Name | rank@1 | rank@5 | lr | weight_decay | lora_r | lora_alpha | lora_dropout | lora_qkv_proj | lora_train_bias | head_scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr9.415043497660484e-05-arcface:s80.0:m1e-06-lora(r32-a96.0-d0.03762089047721775),qkv-thawed() | 0.5426 | 0.6977 | 9.42e-05 | 1.56e-05 | 32 | 96 | 0.03762 | 1 | 0 | 80 |
| 2 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00013761347974879603-arcface:s80.0:m1e-06-lora(r32-a128.0-d0.003477615956990765),qkv-thawed() | 0.5407 | 0.6957 | 1.38e-04 | 2.04e-04 | 32 | 128 | 0.003478 | 1 | 0 | 80 |
| 3 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00013667423760698198-arcface:s80.0:m1e-06-lora(r8-a32.0-d0.04028041879610007),qkv,proj-thawed() | 0.5407 | 0.6899 | 1.37e-04 | 1.70e-06 | 8 | 32 | 0.04028 | 3 | 0 | 80 |
| 4 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr5.882543345421032e-05-arcface:s64.0:m1e-06-lora(r64-a128.0-d0.03713870824785033),qkv,proj-thawed() | 0.5407 | 0.7151 | 5.88e-05 | 2.69e-04 | 64 | 128 | 0.03714 | 3 | 0 | 64 |
| 5 | swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.0001212938206838256-arcface:s32.0:m1e-06-lora(r32-a128.0-d0.05054430617349231),qkv,proj-thawed() | 0.5388 | 0.6996 | 1.21e-04 | 1.02e-04 | 32 | 128 | 0.05054 | 3 | 0 | 32 |

### Hyperparameter Clustering (top 5)
- `lr`: min=5.883e-05, max=0.0001376, std=3.343e-05
- `weight_decay`: min=1.695e-06, max=0.0002695, std=0.0001168
- `lora_r`: min=8, max=64, std=19.92
- `lora_alpha`: min=32, max=128, std=41.72
- `lora_dropout`: min=0.003478, max=0.05054, std=0.0178
- `lora_qkv_proj`: min=1, max=3, std=1.095
- `lora_train_bias`: min=0, max=0, std=0
- `head_scale`: min=32, max=80, std=20.86

### Full Distribution Stats
- `lr`: mean=0.0001127, std=9.424e-05, range=[1.051e-05, 0.0004849]
- `weight_decay`: mean=0.0001244, std=0.0001228, range=[1.106e-06, 0.0004902]
- `lora_r`: mean=29.24, std=19.26, range=[4, 64]
- `lora_alpha`: mean=93.12, std=35.11, range=[16, 128]
- `lora_dropout`: mean=0.05693, std=0.0358, range=[0.003478, 0.1658]
- `lora_qkv_proj`: mean=1.9, std=1, range=[1, 3]
- `lora_train_bias`: mean=0.3, std=0.4606, range=[0, 1]
- `head_scale`: mean=65.28, std=20.43, range=[32, 96]

### Selected Best Config
**Run**: `swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr9.415043497660484e-05-arcface:s80.0:m1e-06-lora(r32-a96.0-d0.03762089047721775),qkv-thawed()` — **rank@1 = 0.5426**

---

## deit3_lora_adaface

- **File**: `wandb_report-runs_sweep-vit_book_deit3_lora_ada.csv`
- **Model**: `deit3_base_patch16_224.fb_in1k`
- **Head**: adaface
- **Total runs**: 100 (including header row)
- **Successful runs**: 100
- **Failed/NaN runs**: 0 (0.0%)

### Swept parameters
`lr`, `head_scale`, `head_margin`, `adaface_h`, `adaface_t_alpha`

### Fixed LoRA parameters
- `lora_r`: 4
- `lora_alpha`: 24
- `lora_dropout`: 0.1
- `lora_qkv_proj`: 1
- `lora_train_bias`: 0
- `weight_decay`: 2e-06

### Top 5 Runs

| Rank | Name | rank@1 | rank@5 | lr | head_scale | head_margin | adaface_h | adaface_t_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0002545950184362033-adaface:s32.0:m0.22741634046252604:h0.21421962489680813:ta0.012527107879321934-lora(r4-a24.0-d0.1),qkv-thawed() | 0.5891 | 0.7248 | 2.55e-04 | 32 | 0.2274 | 0.2142 | 0.01253 |
| 2 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0004891323512820706-adaface:s32.0:m0.2672042348888918:h0.2747733222822908:ta0.029191481142693484-lora(r4-a24.0-d0.1),qkv-thawed() | 0.5814 | 0.7209 | 4.89e-04 | 32 | 0.2672 | 0.2748 | 0.02919 |
| 3 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0002095044036883487-adaface:s32.0:m0.22658120575115617:h0.2187550198620196:ta0.014454486309937852-lora(r4-a24.0-d0.1),qkv-thawed() | 0.5814 | 0.7035 | 2.10e-04 | 32 | 0.2266 | 0.2188 | 0.01445 |
| 4 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.00024255135146485012-adaface:s32.0:m0.24120665101148575:h0.2297702195775767:ta0.010626007719945572-lora(r4-a24.0-d0.1),qkv-thawed() | 0.5795 | 0.7248 | 2.43e-04 | 32 | 0.2412 | 0.2298 | 0.01063 |
| 5 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0002235978125807021-adaface:s32.0:m0.25059575739127016:h0.23333880020400663:ta0.011615703125349267-lora(r4-a24.0-d0.1),qkv-thawed() | 0.5795 | 0.7229 | 2.24e-04 | 32 | 0.2506 | 0.2333 | 0.01162 |

### Hyperparameter Clustering (top 5)
- `lr`: min=0.0002095, max=0.0004891, std=0.000116
- `head_scale`: min=32, max=32, std=0
- `head_margin`: min=0.2266, max=0.2672, std=0.01702
- `adaface_h`: min=0.2142, max=0.2748, std=0.024
- `adaface_t_alpha`: min=0.01063, max=0.02919, std=0.007682

### Full Distribution Stats
- `lr`: mean=0.0002908, std=0.0001282, range=[1.13e-05, 0.0004961]
- `head_scale`: mean=41.76, std=21.92, range=[32, 96]
- `head_margin`: mean=0.2313, std=0.03036, range=[0.2, 0.3966]
- `adaface_h`: mean=0.2823, std=0.07043, range=[0.2026, 0.4459]
- `adaface_t_alpha`: mean=0.01528, std=0.009599, range=[0.005049, 0.04781]

### Selected Best Config
**Run**: `deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0002545950184362033-adaface:s32.0:m0.22741634046252604:h0.21421962489680813:ta0.012527107879321934-lora(r4-a24.0-d0.1),qkv-thawed()` — **rank@1 = 0.5891**

---

## deit3_lora_arcface

- **File**: `wandb_report-runs_sweep-vit_book_deit3_lora.csv`
- **Model**: `deit3_base_patch16_224.fb_in1k`
- **Head**: arcface
- **Total runs**: 65 (including header row)
- **Successful runs**: 64
- **Failed/NaN runs**: 1 (1.5%)

### Swept parameters
`lr`, `weight_decay`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_qkv_proj`, `lora_train_bias`, `head_scale`

### Top 5 Runs

| Rank | Name | rank@1 | rank@5 | lr | weight_decay | lora_r | lora_alpha | lora_dropout | lora_qkv_proj | lora_train_bias | head_scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.00019385005895182297-arcface:s32.0:m1e-05-lora(r4-a32.0-d0.09733099843815056),qkv,train_bias-thawed() | 0.5659 | 0.7054 | 1.94e-04 | 1.33e-06 | 4 | 32 | 0.09733 | 1 | 1 | 32 |
| 2 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.000220540548901336-arcface:s32.0:m1e-05-lora(r4-a16.0-d0.06336214755071766),qkv,train_bias-thawed() | 0.5601 | 0.7016 | 2.21e-04 | 1.18e-06 | 4 | 16 | 0.06336 | 1 | 1 | 32 |
| 3 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.00012398938312518752-arcface:s32.0:m1e-05-lora(r4-a32.0-d0.06295276444678925),qkv-thawed() | 0.5562 | 0.6919 | 1.24e-04 | 1.51e-06 | 4 | 32 | 0.06295 | 1 | 0 | 32 |
| 4 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0004573314737803363-arcface:s32.0:m1e-05-lora(r4-a32.0-d0.16568592807560323),qkv,train_bias-thawed() | 0.5543 | 0.7151 | 4.57e-04 | 3.60e-06 | 4 | 32 | 0.1657 | 1 | 1 | 32 |
| 5 | deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.00017350753085394136-arcface:s32.0:m1e-05-lora(r4-a16.0-d0.17914526557602617),qkv,train_bias-thawed() | 0.5543 | 0.7151 | 1.74e-04 | 1.24e-06 | 4 | 16 | 0.1791 | 1 | 1 | 32 |

### Hyperparameter Clustering (top 5)
- `lr`: min=0.000124, max=0.0004573, std=0.0001298
- `weight_decay`: min=1.177e-06, max=3.6e-06, std=1.03e-06
- `lora_r`: min=4, max=4, std=0
- `lora_alpha`: min=16, max=32, std=8.764
- `lora_dropout`: min=0.06295, max=0.1791, std=0.05559
- `lora_qkv_proj`: min=1, max=1, std=0
- `lora_train_bias`: min=0, max=1, std=0.4472
- `head_scale`: min=32, max=32, std=0

### Full Distribution Stats
- `lr`: mean=0.0002332, std=0.0001347, range=[1.976e-05, 0.000485]
- `weight_decay`: mean=2.98e-05, std=5.726e-05, range=[1.047e-06, 0.0002813]
- `lora_r`: mean=25.31, std=31.91, range=[4, 96]
- `lora_alpha`: mean=37.75, std=29.26, range=[16, 128]
- `lora_dropout`: mean=0.1121, std=0.05324, range=[0.0003651, 0.1921]
- `lora_qkv_proj`: mean=1.438, std=0.8333, range=[1, 3]
- `lora_train_bias`: mean=0.5469, std=0.5017, range=[0, 1]
- `head_scale`: mean=61.25, std=27.87, range=[32, 96]

### Selected Best Config
**Run**: `deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.00019385005895182297-arcface:s32.0:m1e-05-lora(r4-a32.0-d0.09733099843815056),qkv,train_bias-thawed()` — **rank@1 = 0.5659**

---

## Cross-Sweep Comparison

| Sweep | Head | Best rank@1 | Best Run |
| --- | --- | --- | --- |
| swin_lora_adaface | adaface | 0.5814 | `swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr0.00019313294723680936-adaface:s96.0:m0.30762863249271644:h0.3206180757713378:ta0.023366334755827948-lora(r24-a120.0-d0.05),qkv-thawed()` |
| swin_lora_arcface | arcface | 0.5426 | `swin_base_patch4_window7_224.ms_in1k-img96-bs8-lr9.415043497660484e-05-arcface:s80.0:m1e-06-lora(r32-a96.0-d0.03762089047721775),qkv-thawed()` |
| deit3_lora_adaface | adaface | 0.5891 | `deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.0002545950184362033-adaface:s32.0:m0.22741634046252604:h0.21421962489680813:ta0.012527107879321934-lora(r4-a24.0-d0.1),qkv-thawed()` |
| deit3_lora_arcface | arcface | 0.5659 | `deit3_base_patch16_224.fb_in1k-img96-bs8-lr0.00019385005895182297-arcface:s32.0:m1e-05-lora(r4-a32.0-d0.09733099843815056),qkv,train_bias-thawed()` |

### Per-Family Best

**SWIN family winner**: `swin_lora_adaface` (rank@1 = 0.5814)
- vs `swin_lora_arcface`: +0.0388 rank@1

**DEIT3 family winner**: `deit3_lora_adaface` (rank@1 = 0.5891)
- vs `deit3_lora_arcface`: +0.0232 rank@1

### Instability Observations

- **swin_lora_adaface**: 0/100 runs failed (0%). AdaFace sweeps show low failure rate.
- **swin_lora_arcface**: 0/100 runs failed (0%).
- **deit3_lora_adaface**: 0/100 runs failed (0%). AdaFace sweeps show low failure rate.
- **deit3_lora_arcface**: 1/65 runs failed (2%).


---
*Configs saved to `configs/best_lora_configs.json`*