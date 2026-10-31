# Revalidation report

Sources: results/preliminary, results/baselines. Every checkpoint was reloaded and scored again on the CPU
with a NumPy evaluator that shares no code with the training loop. Errors are on the standardised
scale used by the published tables. Exact means cover every test window; drop-last means keep the
first floor(n / batch) * batch windows, the protocol behind the published numbers.

## 1. Agreement between records and recomputation

| run | device | recorded test MSE | recomputed | max relative gap | checkpoint SHA-256 | status |
|---|---|---|---|---|---|---|
| dlinear_ETTh1_h96_s2021 | cpu | 0.371556 | 0.371556 | 7.8e-10 | b12b0a551901 | pass |
| dlinear_ETTh1_h96_s2022 | cpu | 0.375453 | 0.375453 | 6.6e-10 | 652a5cda1b9c | pass |
| dlinear_ETTh1_h96_s2023 | cpu | 0.396269 | 0.396269 | 5.0e-10 | c7254853fdad | pass |
| dlinear_ILI_h24_s2021 | cpu | 2.330591 | 2.330591 | 1.0e-08 | 8257e6a996e8 | pass |
| dlinear_ILI_h24_s2022 | cpu | 2.239182 | 2.239182 | 6.2e-09 | 994b80e92b5d | pass |
| dlinear_ILI_h24_s2023 | cpu | 2.240954 | 2.240954 | 9.5e-09 | 006c010a3215 | pass |
| dlinear_ILI_h36_s2021 | cpu | 2.251219 | 2.251219 | 8.5e-09 | 5ee655cca62b | pass |
| dlinear_ILI_h36_s2022 | cpu | 2.332449 | 2.332449 | 9.1e-09 | 7d92cd918aa8 | pass |
| dlinear_ILI_h36_s2023 | cpu | 2.314276 | 2.314276 | 4.4e-09 | ab929bd667dd | pass |
| dlinear_ILI_h48_s2021 | cpu | 2.361976 | 2.361976 | 4.9e-09 | 082955280b09 | pass |
| dlinear_ILI_h48_s2022 | cpu | 2.270271 | 2.270271 | 1.2e-08 | 9e3e5135f399 | pass |
| dlinear_ILI_h48_s2023 | cpu | 2.290708 | 2.290708 | 2.3e-09 | 567c0f69b4db | pass |
| dlinear_ILI_h60_s2021 | cpu | 2.451976 | 2.451976 | 1.1e-08 | b492eabf1aaf | pass |
| dlinear_ILI_h60_s2022 | cpu | 2.546297 | 2.546297 | 3.1e-09 | 6fb7ff4e7e9f | pass |
| dlinear_ILI_h60_s2023 | cpu | 2.404911 | 2.404911 | 9.7e-09 | 3f5b97b9da4f | pass |
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
| dlinear_ETTh1_h192_s2021 | cpu | 0.418640 | 0.418640 | 8.5e-10 | 1439b0551426 | pass |
| dlinear_ETTh1_h192_s2022 | cpu | 0.408122 | 0.408122 | 8.4e-10 | 327d2f3b5a3f | pass |
| dlinear_ETTh1_h192_s2023 | cpu | 0.498438 | 0.498438 | 7.8e-10 | 838ad2c53f0a | pass |
| dlinear_ETTh1_h336_s2021 | cpu | 0.439766 | 0.439766 | 2.4e-09 | 24075191b149 | pass |
| dlinear_ETTh1_h336_s2022 | cpu | 0.442083 | 0.442083 | 1.1e-09 | bf6bfe64f3fc | pass |
| dlinear_ETTh1_h336_s2023 | cpu | 0.444753 | 0.444753 | 1.4e-09 | 2015c5bc4626 | pass |
| dlinear_ETTh1_h720_s2021 | cpu | 0.478553 | 0.478553 | 1.4e-09 | 8fc7e99a973f | pass |
| dlinear_ETTh1_h720_s2022 | cpu | 0.578812 | 0.578812 | 2.0e-09 | fca75cff345c | pass |
| dlinear_ETTh1_h720_s2023 | cpu | 0.535965 | 0.535965 | 1.2e-09 | e8eb5bbb5bdd | pass |
| patchtst_ETTh1_h192_s2021 | mps | 0.413590 | 0.413590 | 3.8e-09 | 2b16b3d7cdcc | pass |
| patchtst_ETTh1_h192_s2022 | mps | 0.411711 | 0.411711 | 2.1e-09 | 503a6fcff074 | pass |
| patchtst_ETTh1_h192_s2023 | mps | 0.410106 | 0.410106 | 3.5e-09 | ee2b490c906d | pass |
| patchtst_ETTh1_h336_s2021 | mps | 0.441560 | 0.441560 | 2.2e-09 | d6088b95e376 | pass |
| patchtst_ETTh1_h336_s2022 | mps | 0.443839 | 0.443839 | 4.9e-09 | 64c35d6c2d57 | pass |

## 2. Test error per setting

Mean ± sample standard deviation over seeds. The 95% interval is a circular block bootstrap over
test windows of the seed-averaged error, with blocks one horizon long.

| model | dataset | T | params | test MSE exact | 95% interval | test MAE exact | MSE drop-last | seed 2021 drop-last | published | gap of mean | gap of seed 2021 | published within 1 sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 64,704 | 0.3811 ± 0.0133 | 0.3311 to 0.4396 | 0.4050 ± 0.0159 | 0.3810 ± 0.0133 | 0.3715 | 0.375 | +1.6% | -0.9% | yes |
| dlinear | ETTh1 | 192 | 129,408 | 0.4417 ± 0.0494 | 0.3830 to 0.5050 | 0.4475 ± 0.0427 | 0.4417 ± 0.0494 | 0.4186 | 0.405 | +9.1% | +3.4% | yes |
| dlinear | ETTh1 | 336 | 226,464 | 0.4422 ± 0.0025 | 0.3633 to 0.5199 | 0.4441 ± 0.0033 | 0.4422 ± 0.0025 | 0.4397 | 0.439 | +0.7% | +0.2% | no |
| dlinear | ETTh1 | 720 | 485,280 | 0.5311 ± 0.0503 | 0.4588 to 0.6127 | 0.5301 ± 0.0353 | 0.5301 ± 0.0502 | 0.4776 | 0.472 | +12.3% | +1.2% | no |
| dlinear | ILI | 24 | 5,040 | 2.2702 ± 0.0523 | 1.3909 to 3.2757 | 1.0584 ± 0.0281 | 2.0054 ± 0.0347 | 2.0438 | 2.215 | -9.5% | -7.7% | no |
| dlinear | ILI | 36 | 7,560 | 2.2993 ± 0.0426 | 1.4679 to 3.1857 | 1.0769 ± 0.0168 | 2.1312 ± 0.0461 | 2.0796 | 1.963 | +8.6% | +5.9% | no |
| dlinear | ILI | 48 | 10,080 | 2.3077 ± 0.0481 | 1.3606 to 3.2329 | 1.0836 ± 0.0156 | 2.1368 ± 0.0103 | 2.1387 | 2.130 | +0.3% | +0.4% | yes |
| dlinear | ILI | 60 | 12,600 | 2.4677 ± 0.0720 | 1.6512 to 3.3088 | 1.1265 ± 0.0184 | 2.3877 ± 0.0759 | 2.3707 | 2.368 | +0.8% | +0.1% | yes |
| patchtst | ETTh1 | 96 | 81,728 | 0.3775 ± 0.0004 | 0.3260 to 0.4349 | 0.3995 ± 0.0002 | 0.3737 ± 0.0004 | 0.3738 | 0.375 | -0.3% | -0.3% | no |
| patchtst | ETTh1 | 192 | 146,336 | 0.4118 ± 0.0017 | 0.3519 to 0.4772 | 0.4196 ± 0.0010 | 0.4118 ± 0.0017 | 0.4135 | 0.414 | -0.5% | -0.1% | no |
| patchtst | ETTh1 | 336 | 243,248 | 0.4427 ± 0.0016 | 0.3624 to 0.5248 | 0.4428 ± 0.0019 | 0.4322 ± 0.0008 | 0.4316 | 0.431 | +0.3% | +0.1% | no |
| patchtst | ILI | 24 | 33,400 | 2.0350 ± 0.1237 | 1.0517 to 3.3100 | 0.8978 ± 0.0560 | 1.5652 ± 0.1991 | 1.7737 | 1.522 | +2.8% | +16.5% | yes |
| patchtst | ILI | 36 | 41,476 | 1.8984 ± 0.0851 | 1.0098 to 2.8443 | 0.8948 ± 0.0382 | 1.4921 ± 0.2218 | 1.7460 | 1.430 | +4.3% | +22.1% | yes |
| patchtst | ILI | 48 | 49,552 | 1.8245 ± 0.1641 | 0.9363 to 2.7398 | 0.8957 ± 0.0799 | 1.7564 ± 0.1696 | 1.7075 | 1.673 | +5.0% | +2.1% | yes |
| patchtst | ILI | 60 | 57,628 | 1.8790 ± 0.0394 | 1.1611 to 2.6277 | 0.9437 ± 0.0194 | 1.7529 ± 0.0415 | 1.7807 | 1.529 | +14.6% | +16.5% | no |

## 3. Learned models against naive forecasts (exact test MSE)

Repeat-last copies the final input step over the horizon; seasonal repeats the last day (ETTh1) or
the last 52 weeks (ILI).

| model | dataset | T | model MSE | repeat-last MSE | seasonal MSE | model / best naive |
|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 0.3811 | 1.2944 | 0.5122 | 0.74 |
| dlinear | ETTh1 | 192 | 0.4417 | 1.3249 | 0.5808 | 0.76 |
| dlinear | ETTh1 | 336 | 0.4422 | 1.3299 | 0.6499 | 0.68 |
| dlinear | ETTh1 | 720 | 0.5311 | 1.3351 | 0.6554 | 0.81 |
| dlinear | ILI | 24 | 2.2702 | 6.2133 | 2.5638 | 0.89 |
| dlinear | ILI | 36 | 2.2993 | 7.7138 | 2.2655 | 1.01 |
| dlinear | ILI | 48 | 2.3077 | 7.8513 | 2.1425 | 1.08 |
| dlinear | ILI | 60 | 2.4677 | 6.8849 | 1.9174 | 1.29 |
| patchtst | ETTh1 | 96 | 0.3775 | 1.2944 | 0.5122 | 0.74 |
| patchtst | ETTh1 | 192 | 0.4118 | 1.3249 | 0.5808 | 0.71 |
| patchtst | ETTh1 | 336 | 0.4427 | 1.3299 | 0.6499 | 0.68 |
| patchtst | ILI | 24 | 2.0350 | 6.2133 | 2.5638 | 0.79 |
| patchtst | ILI | 36 | 1.8984 | 7.7138 | 2.2655 | 0.84 |
| patchtst | ILI | 48 | 1.8245 | 7.8513 | 2.1425 | 0.85 |
| patchtst | ILI | 60 | 1.8790 | 6.8849 | 1.9174 | 0.98 |

## 4. Error by forecast step (exact test MSE, seed mean)

| model | dataset | T | steps | MSE at those steps |
|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 1, 24, 48, 72, 96 | 0.1032, 0.3279, 0.3689, 0.4093, 0.4225 |
| dlinear | ETTh1 | 192 | 1, 48, 96, 144, 192 | 0.1146, 0.3633, 0.4295, 0.4808, 0.4902 |
| dlinear | ETTh1 | 336 | 1, 84, 168, 252, 336 | 0.1073, 0.4169, 0.4510, 0.4853, 0.4875 |
| dlinear | ETTh1 | 720 | 1, 180, 360, 540, 720 | 0.1168, 0.4563, 0.4841, 0.6191, 0.6965 |
| dlinear | ILI | 24 | 1, 6, 12, 18, 24 | 0.2410, 1.5111, 2.4335, 2.9569, 3.3055 |
| dlinear | ILI | 36 | 1, 9, 18, 27, 36 | 0.3088, 2.1114, 2.4874, 2.8098, 3.1080 |
| dlinear | ILI | 48 | 1, 12, 24, 36, 48 | 0.3068, 2.5265, 2.4115, 2.1448, 3.2101 |
| dlinear | ILI | 60 | 1, 15, 30, 45, 60 | 0.2386, 2.9331, 2.2364, 2.4187, 3.8216 |
| patchtst | ETTh1 | 96 | 1, 24, 48, 72, 96 | 0.1407, 0.3268, 0.3569, 0.3874, 0.4152 |
| patchtst | ETTh1 | 192 | 1, 48, 96, 144, 192 | 0.1469, 0.3516, 0.4149, 0.4444, 0.4579 |
| patchtst | ETTh1 | 336 | 1, 84, 168, 252, 336 | 0.2009, 0.4329, 0.4497, 0.4770, 0.4628 |
| patchtst | ILI | 24 | 1, 6, 12, 18, 24 | 0.5520, 1.3973, 1.8983, 2.7920, 3.2240 |
| patchtst | ILI | 36 | 1, 9, 18, 27, 36 | 0.5361, 1.6706, 1.5927, 2.3373, 2.9828 |
| patchtst | ILI | 48 | 1, 12, 24, 36, 48 | 0.5951, 1.5854, 1.6676, 1.7955, 3.2999 |
| patchtst | ILI | 60 | 1, 15, 30, 45, 60 | 0.6194, 1.9375, 1.5297, 1.7668, 3.4464 |

## 5. Error by variable on ETTh1 (exact test MSE, seed mean)

| model | T | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
|---|---|---|---|---|---|---|---|---|
| dlinear | 96 | 0.755 | 0.213 | 0.780 | 0.173 | 0.516 | 0.124 | 0.107 |
| dlinear | 192 | 0.857 | 0.254 | 0.884 | 0.203 | 0.569 | 0.144 | 0.182 |
| dlinear | 336 | 0.854 | 0.254 | 0.871 | 0.202 | 0.596 | 0.149 | 0.170 |
| dlinear | 720 | 0.861 | 0.324 | 0.854 | 0.255 | 0.800 | 0.187 | 0.436 |
| patchtst | 96 | 0.780 | 0.211 | 0.802 | 0.172 | 0.506 | 0.119 | 0.053 |
| patchtst | 192 | 0.853 | 0.235 | 0.874 | 0.188 | 0.535 | 0.131 | 0.068 |
| patchtst | 336 | 0.912 | 0.248 | 0.926 | 0.198 | 0.588 | 0.148 | 0.079 |

## 5. Error by variable on ILI (exact test MSE, seed mean)

| model | T | % WEIGHTED ILI | %UNWEIGHTED ILI | AGE 0-4 | AGE 5-24 | ILITOTAL | NUM. OF PROVIDERS | OT |
|---|---|---|---|---|---|---|---|---|
| dlinear | 24 | 0.941 | 1.222 | 3.420 | 2.941 | 4.234 | 1.323 | 1.810 |
| dlinear | 36 | 0.838 | 1.114 | 3.320 | 2.825 | 4.266 | 1.583 | 2.149 |
| dlinear | 48 | 0.788 | 1.064 | 3.392 | 2.770 | 4.304 | 1.620 | 2.217 |
| dlinear | 60 | 0.785 | 1.066 | 3.538 | 2.893 | 4.655 | 1.813 | 2.523 |
| patchtst | 24 | 0.987 | 1.303 | 3.486 | 2.950 | 4.113 | 0.378 | 1.029 |
| patchtst | 36 | 0.864 | 1.150 | 3.066 | 2.744 | 3.766 | 0.622 | 1.077 |
| patchtst | 48 | 0.768 | 1.036 | 3.009 | 2.712 | 3.642 | 0.638 | 0.966 |
| patchtst | 60 | 0.750 | 1.016 | 2.941 | 2.706 | 3.814 | 0.803 | 1.123 |

## 6. PatchTST/42 minus DLinear, paired over test windows

Negative means PatchTST is better. The interval is the block bootstrap of the mean difference.

| dataset | T | seeds | mean difference | 95% interval | windows where PatchTST is better |
|---|---|---|---|---|---|
| ETTh1 | 96 | 3 | -0.0036 | -0.0120 to +0.0053 | 56% |
| ETTh1 | 192 | 3 | -0.0299 | -0.0439 to -0.0111 | 83% |
| ETTh1 | 336 | 2 | +0.0018 | -0.0184 to +0.0274 | 55% |
| ILI | 24 | 3 | -0.2352 | -0.7057 to +0.3267 | 82% |
| ILI | 36 | 3 | -0.4009 | -0.9266 to +0.0906 | 87% |
| ILI | 48 | 3 | -0.4832 | -0.8622 to -0.1577 | 84% |
| ILI | 60 | 3 | -0.5888 | -0.8578 to -0.3443 | 93% |

## 7. Windows removed by the drop-last protocol

| model | dataset | T | windows | kept | MSE of kept windows | MSE of dropped windows |
|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 2785 | 2784 | 0.3810 | 0.5223 |
| dlinear | ETTh1 | 192 | 2689 | 2688 | 0.4417 | 0.5346 |
| dlinear | ETTh1 | 336 | 2545 | 2528 | 0.4422 | 0.4483 |
| dlinear | ETTh1 | 720 | 2161 | 2144 | 0.5301 | 0.6531 |
| dlinear | ILI | 24 | 170 | 160 | 2.0054 | 6.5077 |
| dlinear | ILI | 36 | 158 | 128 | 2.1312 | 3.0164 |
| dlinear | ILI | 48 | 146 | 128 | 2.1368 | 3.5224 |
| dlinear | ILI | 60 | 134 | 128 | 2.3877 | 4.1743 |
| patchtst | ETTh1 | 96 | 2785 | 2688 | 0.3737 | 0.4822 |
| patchtst | ETTh1 | 192 | 2689 | 2688 | 0.4118 | 0.5341 |
| patchtst | ETTh1 | 336 | 2545 | 2432 | 0.4322 | 0.6695 |
| patchtst | ILI | 24 | 170 | 160 | 1.5652 | 9.5530 |
| patchtst | ILI | 36 | 158 | 144 | 1.4921 | 6.0780 |
| patchtst | ILI | 48 | 146 | 144 | 1.7564 | 6.7260 |
| patchtst | ILI | 60 | 134 | 128 | 1.7529 | 4.5693 |

## 8. Training

| model | dataset | T | devices | best epoch per seed | epochs run | seconds per epoch | validation MSE |
|---|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | cpu | 8, 6, 4 | 10, 9, 7 | 0.4 ± 0.1 | 0.6425 ± 0.0056 |
| dlinear | ETTh1 | 192 | cpu | 5, 6, 1 | 8, 9, 4 | 1.3 ± 0.1 | 0.8640 ± 0.0078 |
| dlinear | ETTh1 | 336 | cpu | 6, 7, 5 | 9, 10, 8 | 1.6 ± 0.1 | 1.0462 ± 0.0064 |
| dlinear | ETTh1 | 720 | cpu | 6, 1, 3 | 9, 4, 6 | 2.2 ± 0.0 | 1.2294 ± 0.0246 |
| dlinear | ILI | 24 | cpu | 3, 6, 7 | 6, 9, 10 | 0.1 ± 0.0 | 0.2841 ± 0.0110 |
| dlinear | ILI | 36 | cpu | 4, 4, 3 | 7, 7, 6 | 0.1 ± 0.0 | 0.2581 ± 0.0017 |
| dlinear | ILI | 48 | cpu | 3, 5, 6 | 6, 8, 9 | 0.1 ± 0.0 | 0.2420 ± 0.0104 |
| dlinear | ILI | 60 | cpu | 5, 3, 5 | 8, 6, 8 | 0.1 ± 0.0 | 0.2618 ± 0.0041 |
| patchtst | ETTh1 | 96 | mps | 68, 75, 81 | 100, 100, 100 | 10.2 ± 2.6 | 0.6748 ± 0.0024 |
| patchtst | ETTh1 | 192 | mps | 21, 26, 94 | 100, 100, 100 | 6.9 ± 0.0 | 0.9219 ± 0.0028 |
| patchtst | ETTh1 | 336 | mps | 7, 5 | 100, 100 | 10.1 ± 1.9 | 1.1661 ± 0.0030 |
| patchtst | ILI | 24 | cpu | 6, 7, 10 | 100, 100, 100 | 6.2 ± 0.0 | 0.2193 ± 0.0131 |
| patchtst | ILI | 36 | cpu | 3, 11, 9 | 100, 100, 100 | 5.6 ± 0.0 | 0.2161 ± 0.0075 |
| patchtst | ILI | 48 | cpu | 6, 21, 7 | 100, 100, 100 | 5.9 ± 0.0 | 0.2112 ± 0.0070 |
| patchtst | ILI | 60 | cpu | 6, 7, 6 | 100, 100, 100 | 6.6 ± 0.0 | 0.2175 ± 0.0128 |
