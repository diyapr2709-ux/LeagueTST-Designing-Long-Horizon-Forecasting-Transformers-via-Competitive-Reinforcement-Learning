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
| patchtst_ETTh1_h96_s2021 | mps | 0.379058 | 0.379058 | 6.8e-09 | 35b2808ec5dc | pass |
| patchtst_ETTh1_h96_s2022 | mps | 0.379796 | 0.379796 | 1.8e-09 | 06004342d417 | pass |
| patchtst_ETTh1_h96_s2023 | mps | 0.378538 | 0.378538 | 8.6e-09 | 1a5a23d7e4f1 | pass |
| patchtst_ILI_h24_s2021 | cpu | 1.900389 | 1.900389 | 1.1e-08 | 8ae5f8182720 | pass |
| patchtst_ILI_h24_s2022 | cpu | 1.961672 | 1.961672 | 7.4e-09 | 6723e2bd91af | pass |
| patchtst_ILI_h24_s2023 | cpu | 1.924666 | 1.924666 | 4.1e-09 | fc9ee81579ec | pass |
| patchtst_ILI_h36_s2021 | cpu | 1.833032 | 1.833032 | 6.9e-09 | 5bb5db836b64 | pass |
| patchtst_ILI_h36_s2022 | cpu | 1.904595 | 1.904595 | 3.5e-09 | ad1a66bc7508 | pass |
| patchtst_ILI_h36_s2023 | cpu | 1.739636 | 1.739636 | 1.5e-08 | 89af4b260d1a | pass |
| patchtst_ILI_h48_s2021 | cpu | 1.825729 | 1.825729 | 6.8e-09 | 7bd6612cf4ae | pass |
| patchtst_ILI_h48_s2022 | cpu | 1.830788 | 1.830788 | 7.0e-09 | 519ea8593617 | pass |
| patchtst_ILI_h48_s2023 | cpu | 1.653123 | 1.653123 | 1.0e-08 | 92d536c7893b | pass |
| patchtst_ILI_h60_s2021 | cpu | 2.017360 | 2.017360 | 1.8e-08 | bbe465eb3e15 | pass |
| patchtst_ILI_h60_s2022 | cpu | 1.667459 | 1.667459 | 1.6e-09 | 3086b9c3d310 | pass |
| patchtst_ILI_h60_s2023 | cpu | 1.709382 | 1.709382 | 5.8e-09 | cdb761b451cf | pass |
| dlinear_ETTh1_h192_s2021 | cpu | 0.418640 | 0.418640 | 8.5e-10 | 1439b0551426 | pass |
| dlinear_ETTh1_h192_s2022 | cpu | 0.408122 | 0.408122 | 8.4e-10 | 327d2f3b5a3f | pass |
| dlinear_ETTh1_h192_s2023 | cpu | 0.498438 | 0.498438 | 7.8e-10 | 838ad2c53f0a | pass |
| dlinear_ETTh1_h336_s2021 | cpu | 0.439766 | 0.439766 | 2.4e-09 | 24075191b149 | pass |
| dlinear_ETTh1_h336_s2022 | cpu | 0.442083 | 0.442083 | 1.1e-09 | bf6bfe64f3fc | pass |
| dlinear_ETTh1_h336_s2023 | cpu | 0.444753 | 0.444753 | 1.4e-09 | 2015c5bc4626 | pass |
| dlinear_ETTh1_h720_s2021 | cpu | 0.478553 | 0.478553 | 1.4e-09 | 8fc7e99a973f | pass |
| dlinear_ETTh1_h720_s2022 | cpu | 0.578812 | 0.578812 | 2.0e-09 | fca75cff345c | pass |
| dlinear_ETTh1_h720_s2023 | cpu | 0.535965 | 0.535965 | 1.2e-09 | e8eb5bbb5bdd | pass |
| patchtst_ETTh1_h192_s2021 | mps | 0.413907 | 0.413907 | 6.6e-09 | 1410b4cf2e25 | pass |
| patchtst_ETTh1_h192_s2022 | mps | 0.413580 | 0.413580 | 4.1e-09 | 73b47dde2df6 | pass |
| patchtst_ETTh1_h192_s2023 | mps | 0.411431 | 0.411431 | 3.4e-09 | 288259d73c8d | pass |
| patchtst_ETTh1_h336_s2021 | mps | 0.444073 | 0.444073 | 7.0e-09 | f197ad707379 | pass |
| patchtst_ETTh1_h336_s2022 | mps | 0.440954 | 0.440954 | 5.2e-09 | 6d977da315b4 | pass |
| patchtst_ETTh1_h336_s2023 | mps | 0.433565 | 0.433565 | 6.3e-09 | 5e08b5dc92f6 | pass |
| patchtst_ETTh1_h720_s2021 | mps | 0.459396 | 0.459396 | 8.9e-09 | 23a2076d21f6 | pass |
| patchtst_ETTh1_h720_s2022 | mps | 0.456560 | 0.456560 | 5.9e-09 | 0c2ff467b065 | pass |
| patchtst_ETTh1_h720_s2023 | mps | 0.449279 | 0.449279 | 6.3e-09 | 20c09393657b | pass |

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
| patchtst | ETTh1 | 96 | 81,728 | 0.3791 ± 0.0006 | 0.3276 to 0.4373 | 0.4009 ± 0.0005 | 0.3754 ± 0.0005 | 0.3753 | 0.375 | +0.1% | +0.1% | yes |
| patchtst | ETTh1 | 192 | 146,336 | 0.4130 ± 0.0013 | 0.3529 to 0.4790 | 0.4205 ± 0.0007 | 0.4129 ± 0.0013 | 0.4139 | 0.414 | -0.3% | -0.0% | yes |
| patchtst | ETTh1 | 336 | 243,248 | 0.4395 ± 0.0054 | 0.3591 to 0.5216 | 0.4400 ± 0.0042 | 0.4293 ± 0.0050 | 0.4336 | 0.431 | -0.4% | +0.6% | yes |
| patchtst | ETTh1 | 720 | 501,680 | 0.4551 ± 0.0052 | 0.4048 to 0.5105 | 0.4704 ± 0.0037 | 0.4483 ± 0.0052 | 0.4527 | 0.449 | -0.2% | +0.8% | yes |
| patchtst | ILI | 24 | 33,400 | 1.9289 ± 0.0309 | 0.9980 to 3.1443 | 0.8872 ± 0.0230 | 1.4867 ± 0.0614 | 1.4681 | 1.522 | -2.3% | -3.5% | yes |
| patchtst | ILI | 36 | 41,476 | 1.8258 ± 0.0827 | 0.9584 to 2.7496 | 0.8785 ± 0.0298 | 1.4086 ± 0.0310 | 1.4291 | 1.430 | -1.5% | -0.1% | yes |
| patchtst | ILI | 48 | 49,552 | 1.7699 ± 0.1011 | 0.9277 to 2.6518 | 0.8717 ± 0.0441 | 1.7090 ± 0.1100 | 1.7705 | 1.673 | +2.2% | +5.8% | yes |
| patchtst | ILI | 60 | 57,628 | 1.7981 ± 0.1911 | 1.0540 to 2.5723 | 0.8908 ± 0.0489 | 1.6239 ± 0.2221 | 1.8804 | 1.529 | +6.2% | +23.0% | yes |

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
| patchtst | ETTh1 | 96 | 0.3791 | 1.2944 | 0.5122 | 0.74 |
| patchtst | ETTh1 | 192 | 0.4130 | 1.3249 | 0.5808 | 0.71 |
| patchtst | ETTh1 | 336 | 0.4395 | 1.3299 | 0.6499 | 0.68 |
| patchtst | ETTh1 | 720 | 0.4551 | 1.3351 | 0.6554 | 0.69 |
| patchtst | ILI | 24 | 1.9289 | 6.2133 | 2.5638 | 0.75 |
| patchtst | ILI | 36 | 1.8258 | 7.7138 | 2.2655 | 0.81 |
| patchtst | ILI | 48 | 1.7699 | 7.8513 | 2.1425 | 0.83 |
| patchtst | ILI | 60 | 1.7981 | 6.8849 | 1.9174 | 0.94 |

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
| patchtst | ETTh1 | 96 | 1, 24, 48, 72, 96 | 0.1462, 0.3283, 0.3577, 0.3883, 0.4162 |
| patchtst | ETTh1 | 192 | 1, 48, 96, 144, 192 | 0.1509, 0.3527, 0.4162, 0.4452, 0.4581 |
| patchtst | ETTh1 | 336 | 1, 84, 168, 252, 336 | 0.1894, 0.4284, 0.4476, 0.4739, 0.4634 |
| patchtst | ETTh1 | 720 | 1, 180, 360, 540, 720 | 0.2260, 0.4397, 0.4232, 0.4891, 0.5892 |
| patchtst | ILI | 24 | 1, 6, 12, 18, 24 | 0.4501, 1.3366, 1.7532, 2.5872, 3.1044 |
| patchtst | ILI | 36 | 1, 9, 18, 27, 36 | 0.5851, 1.4467, 1.5104, 2.4043, 2.9366 |
| patchtst | ILI | 48 | 1, 12, 24, 36, 48 | 0.6566, 1.5386, 1.5894, 1.7804, 3.1961 |
| patchtst | ILI | 60 | 1, 15, 30, 45, 60 | 0.5058, 1.7135, 1.4026, 1.7254, 3.4644 |

## 5. Error by variable on ETTh1 (exact test MSE, seed mean)

| model | T | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
|---|---|---|---|---|---|---|---|---|
| dlinear | 96 | 0.755 | 0.213 | 0.780 | 0.173 | 0.516 | 0.124 | 0.107 |
| dlinear | 192 | 0.857 | 0.254 | 0.884 | 0.203 | 0.569 | 0.144 | 0.182 |
| dlinear | 336 | 0.854 | 0.254 | 0.871 | 0.202 | 0.596 | 0.149 | 0.170 |
| dlinear | 720 | 0.861 | 0.324 | 0.854 | 0.255 | 0.800 | 0.187 | 0.436 |
| patchtst | 96 | 0.784 | 0.212 | 0.806 | 0.172 | 0.508 | 0.119 | 0.053 |
| patchtst | 192 | 0.855 | 0.235 | 0.877 | 0.188 | 0.536 | 0.131 | 0.068 |
| patchtst | 336 | 0.905 | 0.247 | 0.919 | 0.197 | 0.583 | 0.147 | 0.079 |
| patchtst | 720 | 0.881 | 0.255 | 0.873 | 0.204 | 0.697 | 0.176 | 0.099 |

## 5. Error by variable on ILI (exact test MSE, seed mean)

| model | T | % WEIGHTED ILI | %UNWEIGHTED ILI | AGE 0-4 | AGE 5-24 | ILITOTAL | NUM. OF PROVIDERS | OT |
|---|---|---|---|---|---|---|---|---|
| dlinear | 24 | 0.941 | 1.222 | 3.420 | 2.941 | 4.234 | 1.323 | 1.810 |
| dlinear | 36 | 0.838 | 1.114 | 3.320 | 2.825 | 4.266 | 1.583 | 2.149 |
| dlinear | 48 | 0.788 | 1.064 | 3.392 | 2.770 | 4.304 | 1.620 | 2.217 |
| dlinear | 60 | 0.785 | 1.066 | 3.538 | 2.893 | 4.655 | 1.813 | 2.523 |
| patchtst | 24 | 0.924 | 1.207 | 3.213 | 2.761 | 3.889 | 0.428 | 1.080 |
| patchtst | 36 | 0.792 | 1.058 | 3.003 | 2.726 | 3.631 | 0.531 | 1.038 |
| patchtst | 48 | 0.754 | 1.028 | 2.908 | 2.705 | 3.564 | 0.542 | 0.888 |
| patchtst | 60 | 0.709 | 0.973 | 2.807 | 2.884 | 3.636 | 0.623 | 0.953 |

## 6. PatchTST/42 minus DLinear, paired over test windows

Negative means PatchTST is better. The interval is the block bootstrap of the mean difference.

| dataset | T | seeds | mean difference | 95% interval | windows where PatchTST is better |
|---|---|---|---|---|---|
| ETTh1 | 96 | 3 | -0.0020 | -0.0107 to +0.0075 | 54% |
| ETTh1 | 192 | 3 | -0.0288 | -0.0432 to -0.0095 | 83% |
| ETTh1 | 336 | 3 | -0.0027 | -0.0221 to +0.0220 | 62% |
| ETTh1 | 720 | 3 | -0.0760 | -0.1032 to -0.0518 | 98% |
| ILI | 24 | 3 | -0.3413 | -0.8286 to +0.1882 | 81% |
| ILI | 36 | 3 | -0.4736 | -0.9901 to +0.0190 | 89% |
| ILI | 48 | 3 | -0.5378 | -0.9388 to -0.2182 | 83% |
| ILI | 60 | 3 | -0.6697 | -1.0440 to -0.3315 | 92% |

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
| patchtst | ETTh1 | 96 | 2785 | 2688 | 0.3754 | 0.4821 |
| patchtst | ETTh1 | 192 | 2689 | 2688 | 0.4129 | 0.5327 |
| patchtst | ETTh1 | 336 | 2545 | 2432 | 0.4293 | 0.6602 |
| patchtst | ETTh1 | 720 | 2161 | 2048 | 0.4483 | 0.5787 |
| patchtst | ILI | 24 | 170 | 160 | 1.4867 | 9.0048 |
| patchtst | ILI | 36 | 158 | 144 | 1.4086 | 6.1162 |
| patchtst | ILI | 48 | 146 | 144 | 1.7090 | 6.1510 |
| patchtst | ILI | 60 | 134 | 128 | 1.6239 | 5.5126 |

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
| patchtst | ETTh1 | 96 | mps | 68, 75, 81 | 100, 100, 100 | 7.8 ± 2.9 | 0.6772 ± 0.0018 |
| patchtst | ETTh1 | 192 | mps | 89, 26, 94 | 100, 100, 100 | 4.8 ± 0.6 | 0.9231 ± 0.0024 |
| patchtst | ETTh1 | 336 | mps | 8, 9, 17 | 100, 100, 100 | 4.0 ± 0.1 | 1.1661 ± 0.0043 |
| patchtst | ETTh1 | 720 | mps | 6, 5, 7 | 100, 100, 100 | 3.7 ± 0.1 | 1.4454 ± 0.0081 |
| patchtst | ILI | 24 | cpu | 11, 7, 17 | 100, 100, 100 | 6.9 ± 0.0 | 0.2149 ± 0.0066 |
| patchtst | ILI | 36 | cpu | 11, 12, 12 | 100, 100, 100 | 5.5 ± 0.0 | 0.2049 ± 0.0121 |
| patchtst | ILI | 48 | cpu | 5, 13, 12 | 100, 100, 100 | 2.2 ± 0.0 | 0.2073 ± 0.0116 |
| patchtst | ILI | 60 | cpu | 4, 63, 11 | 100, 100, 100 | 3.6 ± 0.0 | 0.2340 ± 0.0146 |
