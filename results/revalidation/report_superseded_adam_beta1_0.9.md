# Revalidation report

Sources: results/superseded/patchtst_adam_beta1_0.9. Every checkpoint was reloaded and scored again on the CPU
with a NumPy evaluator that shares no code with the training loop. Errors are on the standardised
scale used by the published tables. Exact means cover every test window; drop-last means keep the
first floor(n / batch) * batch windows, the protocol behind the published numbers.

## 1. Agreement between records and recomputation

| run | device | recorded test MSE | recomputed | max relative gap | checkpoint SHA-256 | status |
|---|---|---|---|---|---|---|
| patchtst_ETTh1_h192_s2021 | mps | 0.413590 | 0.413590 | 3.8e-09 | 2b16b3d7cdcc | pass |
| patchtst_ETTh1_h192_s2022 | mps | 0.411711 | 0.411711 | 2.1e-09 | 503a6fcff074 | pass |
| patchtst_ETTh1_h192_s2023 | mps | 0.410106 | 0.410106 | 3.5e-09 | ee2b490c906d | pass |
| patchtst_ETTh1_h336_s2021 | mps | 0.441560 | 0.441560 | 2.2e-09 | d6088b95e376 | pass |
| patchtst_ETTh1_h336_s2022 | mps | 0.443839 | 0.443839 | 4.9e-09 | 64c35d6c2d57 | pass |
| patchtst_ETTh1_h336_s2023 | mps | 0.434928 | 0.434928 | 8.9e-09 | e00e9f252c63 | pass |
| patchtst_ETTh1_h96_s2021 | mps | 0.377580 | 0.377580 | 3.5e-09 | c53e1f4d8fa2 | pass |
| patchtst_ETTh1_h96_s2022 | mps | 0.377841 | 0.377841 | 2.7e-09 | 6185cf2a4ba6 | pass |
| patchtst_ETTh1_h96_s2023 | mps | 0.376983 | 0.376983 | 1.0e-09 | f84acb275b05 | pass |
| patchtst_ILI_h24_s2021 | cpu | 2.163270 | 2.163270 | 8.0e-09 | 16733f39228c | pass |
| patchtst_ILI_h24_s2022 | cpu | 2.025470 | 2.025470 | 1.2e-08 | 12632f8e5eeb | pass |
| patchtst_ILI_h24_s2023 | cpu | 1.916409 | 1.916409 | 8.1e-09 | e770773dc784 | pass |
| patchtst_ILI_h36_s2021 | cpu | 1.988628 | 1.988628 | 6.5e-09 | 07ba35cacb12 | pass |
| patchtst_ILI_h36_s2022 | cpu | 1.819662 | 1.819662 | 6.5e-09 | e14ecbd4c66a | pass |
| patchtst_ILI_h36_s2023 | cpu | 1.886968 | 1.886968 | 2.0e-08 | 0d6826fb2960 | pass |
| patchtst_ILI_h48_s2021 | cpu | 1.788211 | 1.788211 | 5.5e-09 | 6cf0d11710af | pass |
| patchtst_ILI_h48_s2022 | cpu | 1.681587 | 1.681587 | 6.6e-09 | 739d4131f6ac | pass |
| patchtst_ILI_h48_s2023 | cpu | 2.003635 | 2.003635 | 1.1e-08 | 40c267cf554f | pass |
| patchtst_ILI_h60_s2021 | cpu | 1.896919 | 1.896919 | 6.8e-09 | 05950743c45d | pass |
| patchtst_ILI_h60_s2022 | cpu | 1.906249 | 1.906249 | 8.9e-09 | ef735081a0a8 | pass |
| patchtst_ILI_h60_s2023 | cpu | 1.833739 | 1.833739 | 6.1e-09 | 65aa95b86fb8 | pass |

## 2. Test error per setting

Mean ± sample standard deviation over seeds. The 95% interval is a circular block bootstrap over
test windows of the seed-averaged error, with blocks one horizon long.

| model | dataset | T | params | test MSE exact | 95% interval | test MAE exact | MSE drop-last | seed 2021 drop-last | published | gap of mean | gap of seed 2021 | published within 1 sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patchtst | ETTh1 | 96 | 81,728 | 0.3775 ± 0.0004 | 0.3245 to 0.4392 | 0.3995 ± 0.0002 | 0.3737 ± 0.0004 | 0.3738 | 0.375 | -0.3% | -0.3% | no |
| patchtst | ETTh1 | 192 | 146,336 | 0.4118 ± 0.0017 | 0.3513 to 0.4728 | 0.4196 ± 0.0010 | 0.4118 ± 0.0017 | 0.4135 | 0.414 | -0.5% | -0.1% | no |
| patchtst | ETTh1 | 336 | 243,248 | 0.4401 ± 0.0046 | 0.3638 to 0.5242 | 0.4407 ± 0.0038 | 0.4298 ± 0.0041 | 0.4316 | 0.431 | -0.3% | +0.1% | yes |
| patchtst | ILI | 24 | 33,400 | 2.0350 ± 0.1237 | 1.0789 to 3.3553 | 0.8978 ± 0.0560 | 1.5652 ± 0.1991 | 1.7737 | 1.522 | +2.8% | +16.5% | yes |
| patchtst | ILI | 36 | 41,476 | 1.8984 ± 0.0851 | 1.0719 to 2.8353 | 0.8948 ± 0.0382 | 1.4921 ± 0.2218 | 1.7460 | 1.430 | +4.3% | +22.1% | yes |
| patchtst | ILI | 48 | 49,552 | 1.8245 ± 0.1641 | 0.9626 to 2.7166 | 0.8957 ± 0.0799 | 1.7564 ± 0.1696 | 1.7075 | 1.673 | +5.0% | +2.1% | yes |
| patchtst | ILI | 60 | 57,628 | 1.8790 ± 0.0394 | 1.1920 to 2.6438 | 0.9437 ± 0.0194 | 1.7529 ± 0.0415 | 1.7807 | 1.529 | +14.6% | +16.5% | no |

## 3. Learned models against naive forecasts (exact test MSE)

Repeat-last copies the final input step over the horizon; seasonal repeats the last day (ETTh1) or
the last 52 weeks (ILI).

| model | dataset | T | model MSE | repeat-last MSE | seasonal MSE | model / best naive |
|---|---|---|---|---|---|---|
| patchtst | ETTh1 | 96 | 0.3775 | 1.2944 | 0.5122 | 0.74 |
| patchtst | ETTh1 | 192 | 0.4118 | 1.3249 | 0.5808 | 0.71 |
| patchtst | ETTh1 | 336 | 0.4401 | 1.3299 | 0.6499 | 0.68 |
| patchtst | ILI | 24 | 2.0350 | 6.2133 | 2.5638 | 0.79 |
| patchtst | ILI | 36 | 1.8984 | 7.7138 | 2.2655 | 0.84 |
| patchtst | ILI | 48 | 1.8245 | 7.8513 | 2.1425 | 0.85 |
| patchtst | ILI | 60 | 1.8790 | 6.8849 | 1.9174 | 0.98 |

## 4. Error by forecast step (exact test MSE, seed mean)

| model | dataset | T | steps | MSE at those steps |
|---|---|---|---|---|
| patchtst | ETTh1 | 96 | 1, 24, 48, 72, 96 | 0.1407, 0.3268, 0.3569, 0.3874, 0.4152 |
| patchtst | ETTh1 | 192 | 1, 48, 96, 144, 192 | 0.1469, 0.3516, 0.4149, 0.4444, 0.4579 |
| patchtst | ETTh1 | 336 | 1, 84, 168, 252, 336 | 0.1930, 0.4275, 0.4482, 0.4750, 0.4636 |
| patchtst | ILI | 24 | 1, 6, 12, 18, 24 | 0.5520, 1.3973, 1.8983, 2.7920, 3.2240 |
| patchtst | ILI | 36 | 1, 9, 18, 27, 36 | 0.5361, 1.6706, 1.5927, 2.3373, 2.9828 |
| patchtst | ILI | 48 | 1, 12, 24, 36, 48 | 0.5951, 1.5854, 1.6676, 1.7955, 3.2999 |
| patchtst | ILI | 60 | 1, 15, 30, 45, 60 | 0.6194, 1.9375, 1.5297, 1.7668, 3.4464 |

## 5. Error by variable on ETTh1 (exact test MSE, seed mean)

| model | T | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
|---|---|---|---|---|---|---|---|---|
| patchtst | 96 | 0.780 | 0.211 | 0.802 | 0.172 | 0.506 | 0.119 | 0.053 |
| patchtst | 192 | 0.853 | 0.235 | 0.874 | 0.188 | 0.535 | 0.131 | 0.068 |
| patchtst | 336 | 0.905 | 0.247 | 0.919 | 0.197 | 0.585 | 0.148 | 0.079 |

## 5. Error by variable on ILI (exact test MSE, seed mean)

| model | T | % WEIGHTED ILI | %UNWEIGHTED ILI | AGE 0-4 | AGE 5-24 | ILITOTAL | NUM. OF PROVIDERS | OT |
|---|---|---|---|---|---|---|---|---|
| patchtst | 24 | 0.987 | 1.303 | 3.486 | 2.950 | 4.113 | 0.378 | 1.029 |
| patchtst | 36 | 0.864 | 1.150 | 3.066 | 2.744 | 3.766 | 0.622 | 1.077 |
| patchtst | 48 | 0.768 | 1.036 | 3.009 | 2.712 | 3.642 | 0.638 | 0.966 |
| patchtst | 60 | 0.750 | 1.016 | 2.941 | 2.706 | 3.814 | 0.803 | 1.123 |

## 6. PatchTST/42 minus DLinear, paired over test windows

Negative means PatchTST is better. The interval is the block bootstrap of the mean difference.

| dataset | T | seeds | mean difference | 95% interval | windows where PatchTST is better |
|---|---|---|---|---|---|

## 7. Windows removed by the drop-last protocol

| model | dataset | T | windows | kept | MSE of kept windows | MSE of dropped windows |
|---|---|---|---|---|---|---|
| patchtst | ETTh1 | 96 | 2785 | 2688 | 0.3737 | 0.4822 |
| patchtst | ETTh1 | 192 | 2689 | 2688 | 0.4118 | 0.5341 |
| patchtst | ETTh1 | 336 | 2545 | 2432 | 0.4298 | 0.6613 |
| patchtst | ILI | 24 | 170 | 160 | 1.5652 | 9.5530 |
| patchtst | ILI | 36 | 158 | 144 | 1.4921 | 6.0780 |
| patchtst | ILI | 48 | 146 | 144 | 1.7564 | 6.7260 |
| patchtst | ILI | 60 | 134 | 128 | 1.7529 | 4.5693 |

## 8. Training

| model | dataset | T | devices | best epoch per seed | epochs run | seconds per epoch | validation MSE |
|---|---|---|---|---|---|---|---|
| patchtst | ETTh1 | 96 | mps | 68, 75, 81 | 100, 100, 100 | 10.2 ± 2.6 | 0.6748 ± 0.0024 |
| patchtst | ETTh1 | 192 | mps | 21, 26, 94 | 100, 100, 100 | 6.9 ± 0.0 | 0.9219 ± 0.0028 |
| patchtst | ETTh1 | 336 | mps | 7, 5, 9 | 100, 100, 100 | 10.5 ± 1.5 | 1.1642 ± 0.0038 |
| patchtst | ILI | 24 | cpu | 6, 7, 10 | 100, 100, 100 | 6.2 ± 0.0 | 0.2193 ± 0.0131 |
| patchtst | ILI | 36 | cpu | 3, 11, 9 | 100, 100, 100 | 5.6 ± 0.0 | 0.2161 ± 0.0075 |
| patchtst | ILI | 48 | cpu | 6, 21, 7 | 100, 100, 100 | 5.9 ± 0.0 | 0.2112 ± 0.0070 |
| patchtst | ILI | 60 | cpu | 6, 7, 6 | 100, 100, 100 | 6.6 ± 0.0 | 0.2175 ± 0.0128 |
