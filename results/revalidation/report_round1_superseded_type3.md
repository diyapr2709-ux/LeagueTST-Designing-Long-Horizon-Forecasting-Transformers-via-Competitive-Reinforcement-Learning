# Revalidation report

Sources: results/superseded/patchtst_ili_type3. Every checkpoint was reloaded and scored again on the CPU
with a NumPy evaluator that shares no code with the training loop. Errors are on the standardised
scale used by the published tables. Exact means cover every test window; drop-last means keep the
first floor(n / batch) * batch windows, the protocol behind the published numbers.

## 1. Agreement between records and recomputation

| run | device | recorded test MSE | recomputed | max relative gap | checkpoint SHA-256 | status |
|---|---|---|---|---|---|---|
| patchtst_ILI_h24_s2021 | cpu | 2.056833 | 2.056833 | 2.1e-09 | 7fa96d8e469a | pass |
| patchtst_ILI_h24_s2022 | cpu | 2.035710 | 2.035710 | 5.4e-09 | 33df5b939665 | pass |
| patchtst_ILI_h24_s2023 | cpu | 1.931856 | 1.931856 | 1.2e-08 | 099dd704e6cb | pass |
| patchtst_ILI_h36_s2021 | cpu | 1.802889 | 1.802889 | 7.7e-09 | 670d4f2b9c42 | pass |
| patchtst_ILI_h36_s2022 | cpu | 2.077726 | 2.077726 | 4.4e-09 | 3a808d30eba4 | pass |
| patchtst_ILI_h36_s2023 | cpu | 1.855987 | 1.855987 | 2.0e-08 | e8f913c18fc2 | pass |
| patchtst_ILI_h48_s2021 | cpu | 1.704759 | 1.704759 | 7.4e-09 | 9ab8a5c103d5 | pass |
| patchtst_ILI_h48_s2022 | cpu | 1.759102 | 1.759102 | 8.4e-09 | 99e348ae40bd | pass |
| patchtst_ILI_h48_s2023 | cpu | 1.965877 | 1.965877 | 5.0e-09 | d2e86f45b96f | pass |
| patchtst_ILI_h60_s2021 | cpu | 1.855918 | 1.855918 | 7.3e-09 | 8515244d225e | pass |
| patchtst_ILI_h60_s2022 | cpu | 1.586417 | 1.586417 | 4.9e-09 | 93687036516f | pass |
| patchtst_ILI_h60_s2023 | cpu | 1.966814 | 1.966814 | 1.1e-08 | df08e45a1263 | pass |

## 2. Test error per setting

Mean ± sample standard deviation over seeds. The 95% interval is a circular block bootstrap over
test windows of the seed-averaged error, with blocks one horizon long.

| model | dataset | T | params | test MSE exact | 95% interval | test MAE exact | MSE drop-last | seed 2021 drop-last | published | gap of mean | gap of seed 2021 | published within 1 sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patchtst | ILI | 24 | 33,400 | 2.0081 ± 0.0669 | 1.0493 to 3.2284 | 0.9007 ± 0.0285 | 1.5631 ± 0.1088 | 1.6467 | 1.522 | +2.7% | +8.2% | yes |
| patchtst | ILI | 36 | 41,476 | 1.9122 ± 0.1458 | 0.9837 to 2.8834 | 0.9078 ± 0.0467 | 1.4939 ± 0.2277 | 1.2961 | 1.430 | +4.5% | -9.4% | yes |
| patchtst | ILI | 48 | 49,552 | 1.8099 ± 0.1378 | 0.9881 to 2.6384 | 0.9027 ± 0.0533 | 1.7513 ± 0.1457 | 1.6366 | 1.673 | +4.7% | -2.2% | yes |
| patchtst | ILI | 60 | 57,628 | 1.8030 ± 0.1956 | 1.1209 to 2.5127 | 0.9043 ± 0.0634 | 1.6753 ± 0.2234 | 1.7347 | 1.529 | +9.6% | +13.5% | yes |

## 3. Learned models against naive forecasts (exact test MSE)

Repeat-last copies the final input step over the horizon; seasonal repeats the last day (ETTh1) or
the last 52 weeks (ILI).

| model | dataset | T | model MSE | repeat-last MSE | seasonal MSE | model / best naive |
|---|---|---|---|---|---|---|
| patchtst | ILI | 24 | 2.0081 | 6.2133 | 2.5638 | 0.78 |
| patchtst | ILI | 36 | 1.9122 | 7.7138 | 2.2655 | 0.84 |
| patchtst | ILI | 48 | 1.8099 | 7.8513 | 2.1425 | 0.84 |
| patchtst | ILI | 60 | 1.8030 | 6.8849 | 1.9174 | 0.94 |

## 4. Error by forecast step (exact test MSE, seed mean)

| model | dataset | T | steps | MSE at those steps |
|---|---|---|---|---|
| patchtst | ILI | 24 | 1, 6, 12, 18, 24 | 0.5429, 1.3294, 1.8956, 2.7118, 3.1576 |
| patchtst | ILI | 36 | 1, 9, 18, 27, 36 | 0.5415, 1.5495, 1.6797, 2.5515, 3.0247 |
| patchtst | ILI | 48 | 1, 12, 24, 36, 48 | 0.5546, 1.6872, 1.6569, 1.7753, 3.2279 |
| patchtst | ILI | 60 | 1, 15, 30, 45, 60 | 0.5516, 1.7787, 1.4625, 1.6635, 3.3553 |

## 5. Error by variable on ILI (exact test MSE, seed mean)

| model | T | % WEIGHTED ILI | %UNWEIGHTED ILI | AGE 0-4 | AGE 5-24 | ILITOTAL | NUM. OF PROVIDERS | OT |
|---|---|---|---|---|---|---|---|---|
| patchtst | 24 | 0.966 | 1.284 | 3.438 | 2.916 | 4.069 | 0.366 | 1.019 |
| patchtst | 36 | 0.838 | 1.111 | 3.259 | 2.801 | 3.715 | 0.587 | 1.074 |
| patchtst | 48 | 0.749 | 1.022 | 2.953 | 2.694 | 3.636 | 0.646 | 0.970 |
| patchtst | 60 | 0.719 | 0.987 | 2.859 | 2.675 | 3.709 | 0.676 | 0.996 |

## 6. PatchTST/42 minus DLinear, paired over test windows

Negative means PatchTST is better. The interval is the block bootstrap of the mean difference.

| dataset | T | seeds | mean difference | 95% interval | windows where PatchTST is better |
|---|---|---|---|---|---|

## 7. Windows removed by the drop-last protocol

| model | dataset | T | windows | kept | MSE of kept windows | MSE of dropped windows |
|---|---|---|---|---|---|---|
| patchtst | ILI | 24 | 170 | 160 | 1.5631 | 9.1289 |
| patchtst | ILI | 36 | 158 | 144 | 1.4939 | 6.2143 |
| patchtst | ILI | 48 | 146 | 144 | 1.7513 | 6.0332 |
| patchtst | ILI | 60 | 134 | 128 | 1.6753 | 4.5285 |

## 8. Training

| model | dataset | T | devices | best epoch per seed | epochs run | seconds per epoch | validation MSE |
|---|---|---|---|---|---|---|---|
| patchtst | ILI | 24 | cpu | 8, 7, 10 | 100, 100, 100 | 5.4 ± 2.7 | 0.2122 ± 0.0152 |
| patchtst | ILI | 36 | cpu | 10, 5, 8 | 100, 100, 100 | 6.5 ± 0.3 | 0.2165 ± 0.0074 |
| patchtst | ILI | 48 | cpu | 12, 13, 7 | 100, 100, 100 | 6.4 ± 0.2 | 0.2083 ± 0.0076 |
| patchtst | ILI | 60 | cpu | 6, 15, 7 | 100, 100, 100 | 6.3 ± 0.2 | 0.2258 ± 0.0129 |
