# Revalidation report

Sources: results/seed_study. Every checkpoint was reloaded and scored again on the CPU
with a NumPy evaluator that shares no code with the training loop. Errors are on the standardised
scale used by the published tables. Exact means cover every test window; drop-last means keep the
first floor(n / batch) * batch windows, the protocol behind the published numbers.

## 1. Agreement between records and recomputation

| run | device | recorded test MSE | recomputed | max relative gap | checkpoint SHA-256 | status |
|---|---|---|---|---|---|---|
| dlinear_ETTh1_h192_s2024 | cpu | 0.408753 | 0.408753 | 1.1e-09 | 088908b9fa02 | pass |
| dlinear_ETTh1_h192_s2025 | cpu | 0.410275 | 0.410275 | 2.6e-09 | c47d3b8111e2 | pass |
| dlinear_ETTh1_h192_s2026 | cpu | 0.451019 | 0.451019 | 1.0e-09 | cc181f9ecc2f | pass |
| dlinear_ETTh1_h192_s2027 | cpu | 0.415062 | 0.415062 | 1.0e-09 | 62efced32557 | pass |
| dlinear_ETTh1_h192_s2028 | cpu | 0.419940 | 0.419940 | 2.2e-10 | c6aa96bab4fc | pass |
| dlinear_ETTh1_h192_s2029 | cpu | 0.409697 | 0.409697 | 1.8e-09 | c546db0b46e2 | pass |
| dlinear_ETTh1_h192_s2030 | cpu | 0.435085 | 0.435085 | 3.2e-09 | b5b05e7558ef | pass |
| dlinear_ETTh1_h336_s2024 | cpu | 0.439246 | 0.439246 | 2.4e-09 | 13961f14e7ce | pass |
| dlinear_ETTh1_h336_s2025 | cpu | 0.443134 | 0.443134 | 1.6e-09 | 133c90786c5b | pass |
| dlinear_ETTh1_h336_s2026 | cpu | 0.441006 | 0.441006 | 1.4e-09 | 041010e1b2dd | pass |
| dlinear_ETTh1_h336_s2027 | cpu | 0.437360 | 0.437360 | 9.0e-10 | 2d289f398156 | pass |
| dlinear_ETTh1_h336_s2028 | cpu | 0.443625 | 0.443625 | 1.9e-09 | 207a7f40559c | pass |
| dlinear_ETTh1_h336_s2029 | cpu | 0.460144 | 0.460144 | 1.5e-09 | a67d7481588d | pass |
| dlinear_ETTh1_h336_s2030 | cpu | 0.452481 | 0.452481 | 2.1e-09 | f5bbbd64d79c | pass |
| dlinear_ETTh1_h720_s2024 | cpu | 0.471133 | 0.471133 | 9.1e-10 | 51d2425689d4 | pass |
| dlinear_ETTh1_h720_s2025 | cpu | 0.511706 | 0.511706 | 5.9e-10 | 08ecb28a9e2a | pass |
| dlinear_ETTh1_h720_s2026 | cpu | 0.476494 | 0.476494 | 1.5e-09 | 1f67337ea5fd | pass |
| dlinear_ETTh1_h720_s2027 | cpu | 0.561990 | 0.561990 | 1.2e-09 | 9d7b570a8a1b | pass |
| dlinear_ETTh1_h720_s2028 | cpu | 0.526709 | 0.526709 | 8.5e-10 | 14c77410e809 | pass |
| dlinear_ETTh1_h720_s2029 | cpu | 0.487632 | 0.487632 | 1.9e-09 | 43b44f7f994e | pass |
| dlinear_ETTh1_h720_s2030 | cpu | 0.500345 | 0.500345 | 2.9e-09 | 6ed096230a42 | pass |
| dlinear_ETTh1_h96_s2024 | cpu | 0.378370 | 0.378370 | 1.9e-09 | 2504836e614b | pass |
| dlinear_ETTh1_h96_s2025 | cpu | 0.370296 | 0.370296 | 1.3e-09 | 3f6cde659954 | pass |
| dlinear_ETTh1_h96_s2026 | cpu | 0.371936 | 0.371936 | 1.8e-09 | 2fbfc7d4fb46 | pass |
| dlinear_ETTh1_h96_s2027 | cpu | 0.371993 | 0.371993 | 1.9e-09 | ab7f8578a833 | pass |
| dlinear_ETTh1_h96_s2028 | cpu | 0.371591 | 0.371591 | 1.9e-09 | 2f8a16a44e6f | pass |
| dlinear_ETTh1_h96_s2029 | cpu | 0.370858 | 0.370858 | 1.7e-09 | c0449dbc7f7d | pass |
| dlinear_ETTh1_h96_s2030 | cpu | 0.381951 | 0.381951 | 1.6e-09 | 40376987d9db | pass |
| dlinear_ILI_h24_s2024 | cpu | 2.240399 | 2.240399 | 1.0e-08 | 820fa5e834db | pass |
| dlinear_ILI_h24_s2025 | cpu | 2.218043 | 2.218043 | 1.7e-08 | 771f0542e7d5 | pass |
| dlinear_ILI_h24_s2026 | cpu | 2.221183 | 2.221183 | 1.2e-08 | bb528317e85b | pass |
| dlinear_ILI_h24_s2027 | cpu | 2.314256 | 2.314256 | 7.5e-09 | 3643234c170c | pass |
| dlinear_ILI_h24_s2028 | cpu | 2.258405 | 2.258405 | 7.2e-09 | f7a77bdb6723 | pass |
| dlinear_ILI_h24_s2029 | cpu | 2.201911 | 2.201911 | 1.7e-08 | 10b8fc094793 | pass |
| dlinear_ILI_h24_s2030 | cpu | 2.206977 | 2.206977 | 5.2e-09 | a2d56dd73485 | pass |
| dlinear_ILI_h36_s2024 | cpu | 2.215135 | 2.215135 | 6.0e-09 | 5bca65c9ef4b | pass |
| dlinear_ILI_h36_s2025 | cpu | 2.486304 | 2.486304 | 8.5e-09 | 1a6a83858063 | pass |
| dlinear_ILI_h36_s2026 | cpu | 2.265739 | 2.265739 | 5.4e-09 | 35c079e908d9 | pass |
| dlinear_ILI_h36_s2027 | cpu | 2.265926 | 2.265926 | 1.2e-08 | 8e16cf320b89 | pass |
| dlinear_ILI_h36_s2028 | cpu | 2.255647 | 2.255647 | 1.4e-08 | 61a602e637fe | pass |
| dlinear_ILI_h36_s2029 | cpu | 2.214271 | 2.214271 | 1.2e-08 | 254d54db2988 | pass |
| dlinear_ILI_h36_s2030 | cpu | 2.157684 | 2.157684 | 6.8e-09 | 797ca2020952 | pass |
| dlinear_ILI_h48_s2024 | cpu | 2.229690 | 2.229690 | 3.8e-09 | 70e22ee6600f | pass |
| dlinear_ILI_h48_s2025 | cpu | 2.283777 | 2.283777 | 5.9e-09 | aad494e16d27 | pass |
| dlinear_ILI_h48_s2026 | cpu | 2.358162 | 2.358162 | 5.6e-09 | 0030d89a13a2 | pass |
| dlinear_ILI_h48_s2027 | cpu | 2.262886 | 2.262886 | 5.8e-09 | a3af946b15ff | pass |
| dlinear_ILI_h48_s2028 | cpu | 2.299512 | 2.299512 | 1.1e-08 | 89c943515868 | pass |
| dlinear_ILI_h48_s2029 | cpu | 2.111510 | 2.111510 | 1.1e-08 | 2a425363bbf1 | pass |
| dlinear_ILI_h48_s2030 | cpu | 2.368187 | 2.368187 | 1.1e-08 | 9769aa2fac03 | pass |
| dlinear_ILI_h60_s2024 | cpu | 2.432531 | 2.432531 | 5.1e-09 | a6daaaf025e1 | pass |
| dlinear_ILI_h60_s2025 | cpu | 2.315514 | 2.315514 | 1.7e-08 | 1649eee106ac | pass |
| dlinear_ILI_h60_s2026 | cpu | 2.290902 | 2.290902 | 1.1e-08 | 11321cc68cd0 | pass |
| dlinear_ILI_h60_s2027 | cpu | 2.419642 | 2.419642 | 1.3e-08 | 8143f3659358 | pass |
| dlinear_ILI_h60_s2028 | cpu | 2.528435 | 2.528435 | 1.2e-08 | 36857b63528f | pass |
| dlinear_ILI_h60_s2029 | cpu | 2.383522 | 2.383522 | 2.2e-08 | 44601eeaedea | pass |
| dlinear_ILI_h60_s2030 | cpu | 2.479058 | 2.479058 | 1.3e-08 | d888410b93f0 | pass |

## 2. Test error per setting

Mean ± sample standard deviation over seeds. The 95% interval is a circular block bootstrap over
test windows of the seed-averaged error, with blocks one horizon long.

| model | dataset | T | params | test MSE exact | 95% interval | test MAE exact | MSE drop-last | seed 2021 drop-last | published | gap of mean | gap of seed 2021 | published within 1 sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 64,704 | 0.3739 ± 0.0045 | 0.3226 to 0.4346 | 0.3961 ± 0.0055 | 0.3738 ± 0.0045 |  | 0.375 | -0.3% |  | yes |
| dlinear | ETTh1 | 192 | 129,408 | 0.4214 ± 0.0160 | 0.3672 to 0.4826 | 0.4311 ± 0.0158 | 0.4214 ± 0.0159 |  | 0.405 | +4.0% |  | no |
| dlinear | ETTh1 | 336 | 226,464 | 0.4453 ± 0.0081 | 0.3705 to 0.5228 | 0.4465 ± 0.0075 | 0.4452 ± 0.0082 |  | 0.439 | +1.4% |  | yes |
| dlinear | ETTh1 | 720 | 485,280 | 0.5051 ± 0.0317 | 0.4379 to 0.5839 | 0.5135 ± 0.0230 | 0.5041 ± 0.0317 |  | 0.472 | +6.8% |  | no |
| dlinear | ILI | 24 | 5,040 | 2.2373 ± 0.0391 | 1.3920 to 3.4040 | 1.0432 ± 0.0183 | 1.9760 ± 0.0455 |  | 2.215 | -10.8% |  | no |
| dlinear | ILI | 36 | 7,560 | 2.2658 ± 0.1046 | 1.4815 to 3.1448 | 1.0663 ± 0.0414 | 2.0953 ± 0.1296 |  | 1.963 | +6.7% |  | no |
| dlinear | ILI | 48 | 10,080 | 2.2734 ± 0.0868 | 1.4117 to 3.1541 | 1.0757 ± 0.0287 | 2.1309 ± 0.0941 |  | 2.130 | +0.0% |  | yes |
| dlinear | ILI | 60 | 12,600 | 2.4071 ± 0.0848 | 1.6409 to 3.2515 | 1.1106 ± 0.0290 | 2.3138 ± 0.0962 |  | 2.368 | -2.3% |  | yes |

## 3. Learned models against naive forecasts (exact test MSE)

Repeat-last copies the final input step over the horizon; seasonal repeats the last day (ETTh1) or
the last 52 weeks (ILI).

| model | dataset | T | model MSE | repeat-last MSE | seasonal MSE | model / best naive |
|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 0.3739 | 1.2944 | 0.5122 | 0.73 |
| dlinear | ETTh1 | 192 | 0.4214 | 1.3249 | 0.5808 | 0.73 |
| dlinear | ETTh1 | 336 | 0.4453 | 1.3299 | 0.6499 | 0.69 |
| dlinear | ETTh1 | 720 | 0.5051 | 1.3351 | 0.6554 | 0.77 |
| dlinear | ILI | 24 | 2.2373 | 6.2133 | 2.5638 | 0.87 |
| dlinear | ILI | 36 | 2.2658 | 7.7138 | 2.2655 | 1.00 |
| dlinear | ILI | 48 | 2.2734 | 7.8513 | 2.1425 | 1.06 |
| dlinear | ILI | 60 | 2.4071 | 6.8849 | 1.9174 | 1.26 |

## 4. Error by forecast step (exact test MSE, seed mean)

| model | dataset | T | steps | MSE at those steps |
|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 1, 24, 48, 72, 96 | 0.1023, 0.3271, 0.3588, 0.3925, 0.4174 |
| dlinear | ETTh1 | 192 | 1, 48, 96, 144, 192 | 0.1062, 0.3605, 0.4282, 0.4532, 0.4676 |
| dlinear | ETTh1 | 336 | 1, 84, 168, 252, 336 | 0.1078, 0.4222, 0.4547, 0.4922, 0.4921 |
| dlinear | ETTh1 | 720 | 1, 180, 360, 540, 720 | 0.1120, 0.4465, 0.4907, 0.5432, 0.6989 |
| dlinear | ILI | 24 | 1, 6, 12, 18, 24 | 0.2506, 1.4839, 2.3838, 2.9536, 3.2986 |
| dlinear | ILI | 36 | 1, 9, 18, 27, 36 | 0.2794, 2.1352, 2.4872, 2.7517, 2.9885 |
| dlinear | ILI | 48 | 1, 12, 24, 36, 48 | 0.3042, 2.3890, 2.4690, 2.2213, 3.1742 |
| dlinear | ILI | 60 | 1, 15, 30, 45, 60 | 0.2828, 2.7824, 2.2027, 2.1821, 3.9290 |

## 5. Error by variable on ETTh1 (exact test MSE, seed mean)

| model | T | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
|---|---|---|---|---|---|---|---|---|
| dlinear | 96 | 0.753 | 0.210 | 0.778 | 0.170 | 0.509 | 0.121 | 0.077 |
| dlinear | 192 | 0.832 | 0.242 | 0.859 | 0.193 | 0.547 | 0.137 | 0.140 |
| dlinear | 336 | 0.860 | 0.256 | 0.878 | 0.203 | 0.597 | 0.150 | 0.173 |
| dlinear | 720 | 0.820 | 0.309 | 0.809 | 0.243 | 0.786 | 0.181 | 0.388 |

## 5. Error by variable on ILI (exact test MSE, seed mean)

| model | T | % WEIGHTED ILI | %UNWEIGHTED ILI | AGE 0-4 | AGE 5-24 | ILITOTAL | NUM. OF PROVIDERS | OT |
|---|---|---|---|---|---|---|---|---|
| dlinear | 24 | 0.940 | 1.223 | 3.403 | 2.916 | 4.193 | 1.263 | 1.723 |
| dlinear | 36 | 0.841 | 1.118 | 3.325 | 2.796 | 4.222 | 1.509 | 2.050 |
| dlinear | 48 | 0.778 | 1.051 | 3.341 | 2.751 | 4.247 | 1.581 | 2.165 |
| dlinear | 60 | 0.784 | 1.062 | 3.510 | 2.874 | 4.563 | 1.691 | 2.366 |

## 6. PatchTST/42 minus DLinear, paired over test windows

Negative means PatchTST is better. The interval is the block bootstrap of the mean difference.

| dataset | T | seeds | mean difference | 95% interval | windows where PatchTST is better |
|---|---|---|---|---|---|

## 7. Windows removed by the drop-last protocol

| model | dataset | T | windows | kept | MSE of kept windows | MSE of dropped windows |
|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | 2785 | 2784 | 0.3738 | 0.4872 |
| dlinear | ETTh1 | 192 | 2689 | 2688 | 0.4214 | 0.5312 |
| dlinear | ETTh1 | 336 | 2545 | 2528 | 0.4452 | 0.4517 |
| dlinear | ETTh1 | 720 | 2161 | 2144 | 0.5041 | 0.6338 |
| dlinear | ILI | 24 | 170 | 160 | 1.9760 | 6.4179 |
| dlinear | ILI | 36 | 158 | 128 | 2.0953 | 2.9934 |
| dlinear | ILI | 48 | 146 | 128 | 2.1309 | 3.2868 |
| dlinear | ILI | 60 | 134 | 128 | 2.3138 | 4.3964 |

## 8. Training

| model | dataset | T | devices | best epoch per seed | epochs run | seconds per epoch | validation MSE |
|---|---|---|---|---|---|---|---|
| dlinear | ETTh1 | 96 | cpu | 6, 10, 9, 8, 8, 9, 5 | 9, 10, 10, 10, 10, 10, 8 | 3.1 ± 0.4 | 0.6461 ± 0.0056 |
| dlinear | ETTh1 | 192 | cpu | 7, 6, 3, 5, 4, 5, 3 | 10, 9, 6, 8, 7, 8, 6 | 3.8 ± 0.4 | 0.8639 ± 0.0024 |
| dlinear | ETTh1 | 336 | cpu | 7, 5, 7, 7, 4, 4, 4 | 10, 8, 10, 10, 7, 7, 7 | 5.5 ± 1.8 | 1.0526 ± 0.0107 |
| dlinear | ETTh1 | 720 | cpu | 10, 4, 6, 1, 4, 5, 4 | 10, 7, 9, 4, 7, 8, 7 | 3.7 ± 0.9 | 1.2115 ± 0.0163 |
| dlinear | ILI | 24 | cpu | 8, 3, 3, 3, 6, 5, 5 | 10, 6, 6, 6, 9, 8, 8 | 0.1 ± 0.0 | 0.2850 ± 0.0047 |
| dlinear | ILI | 36 | cpu | 3, 3, 10, 6, 6, 6, 3 | 6, 6, 10, 9, 9, 9, 6 | 0.1 ± 0.0 | 0.2549 ± 0.0057 |
| dlinear | ILI | 48 | cpu | 4, 3, 4, 7, 7, 4, 5 | 7, 6, 7, 10, 10, 7, 8 | 0.1 ± 0.0 | 0.2453 ± 0.0038 |
| dlinear | ILI | 60 | cpu | 3, 3, 2, 7, 3, 6, 5 | 6, 6, 5, 10, 6, 9, 8 | 0.1 ± 0.0 | 0.2566 ± 0.0044 |
