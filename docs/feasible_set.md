# Feasible set and the ILI count

A configuration is feasible at a setting when its PatchTST parameter count
is at most that of the PatchTST/42 reproduction at the same dataset and
horizon. `leaguetst.search.feasible` computes the count in closed form;
the tests check it against instantiated models, and the verifier checked
all 30,720 configurations at all eight settings on the meta device.

| Setting | Budget | Feasible | Channel-independent | Channel-mixing |
|---|---|---|---|---|
| ETTh1 T=96 | 81,728 | 2,720 | 2,720 | 0 |
| ETTh1 T=192 | 146,336 | 2,752 | 2,752 | 0 |
| ETTh1 T=336 | 243,248 | 2,656 | 2,656 | 0 |
| ETTh1 T=720 | 501,680 | 2,464 | 2,464 | 0 |
| ILI T=24 | 33,400 | 6,144 | 4,800 | 1,344 |
| ILI T=36 | 41,476 | 6,016 | 4,960 | 1,056 |
| ILI T=48 | 49,552 | 6,048 | 5,248 | 800 |
| ILI T=60 | 57,628 | 6,048 | 5,376 | 672 |

The progress report's ILI counts (6,144 / 6,016 / 6,048 / 6,048) include
channel-mixing configurations, which fit on ILI because its flatten head
is small. The audit's channel-independent counts (4,800 / 4,960 / 5,248 /
5,376) are the same sets without them. Both are correct; the report now
says which is which.

2,368 configurations fit all eight settings, all channel-independent.
They are 74 distinct architectures (patch length, stride, layers, width,
feed-forward ratio), each repeated 32 times over heads, dropout and
instance normalisation. Widths are 16 (2,080 configs) and 32 (288). On
ILI T=24, 32 channel-mixing configs land exactly on the 33,400 cap and are
counted as feasible ("at most"); they are not in the 2,368.
