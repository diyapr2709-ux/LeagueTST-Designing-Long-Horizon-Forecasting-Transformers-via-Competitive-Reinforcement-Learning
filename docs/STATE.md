# Build state

Updated by every loop iteration. Newest first.

## 2026-11-23

* E0 built and tested (see docs/validation_log.md, pass 7). Train and
  validation rows are in data/trainval/ with MANIFEST.json and uploaded to
  the leaguetst-data volume in both Modal workspaces.
* K5 pilot launched on Modal (ETTh1 on diya-027, ILI on diyapr). A first
  launch that asked for 8 CPUs with an L4 never got a container in five
  minutes; it was stopped and relaunched with 4 CPUs.
* Reviews: the verifier passed all five baseline checks; the critic's
  review led to amendments A1 to A11 in docs/PLAN.md.
* K0 is still open: the vault hand step (move and encrypt the full CSVs) is
  the owner's.
