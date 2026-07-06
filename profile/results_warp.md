# Warp kernel benchmark record (`profile/bench_warp.py`)

Committed results for the fused-warp kernel (`libfusedwarp`). Append a dated block
per measurement campaign; keep the exact command so runs stay comparable.
(`profile/results/` is gitignored machine output — this file is the curated record.)

---

## 2026-07-06 — baseline + Phase-A fuse-once (`bicubic_pred_wy_pair`)

- Machine: Windows 11, Intel Arrow Lake (Family 6 Model 198, 8P+12E), AVX2
- Compiler: clang-cl (VS-bundled LLVM), warp TU flags `/clang:-O3 /clang:-march=native`
- Command: `env/Scripts/python.exe profile/bench_warp.py --size 2048 --pairs 4 --iters 7 --mode both`
- Bench threads: 1 (single-thread kernel timing); impl0 = scalar reference, impl1 = interior/SIMD path
- Repo state: before = main @ 16cbea5; after = + fuse-once change (this commit)

| run | mode | impl0 ref ms/pair | impl1 opt ms/pair | speedup | opt Gtaps/s | max\|Δ\| |
|---|---|---|---|---|---|---|
| before | bicubic | 179.55 | 115.20 | 1.56× | 2.33 | 6.10e-05 |
| before | lanczos | 219.56 | 123.00 | 1.79× | 3.55 | 9.16e-05 |
| after #1 | bicubic | 180.86 | 114.55 | 1.58× | 2.34 | 6.10e-05 |
| after #1 | lanczos | 204.59 | 115.97 | 1.76× | 3.76 | 9.16e-05 |
| after #2 | bicubic | 183.10 | 115.33 | 1.59× | 2.33 | 6.10e-05 |
| after #2 | lanczos | 209.55 | 117.56 | 1.78× | 3.71 | 9.16e-05 |

**Fuse-once outcome:** outputs verified **bit-identical** (exact `np.array_equal` on all
impl × interp combinations, 512² probe) and max|Δ| lines unchanged — the change is purely
structural. Speed: **lanczos ≈ −5% ms/pair** (123.0 → 116.8 avg), **bicubic unchanged
within run noise** (115.2 → 114.9 avg, ±0.7%) — clang's CSE was evidently already
eliminating most of the duplicated Phase-A geometry across the two inlined
`bicubic_pred_wy` calls in the bicubic path; the win materialised only where register
pressure had blocked that (lanczos). The 10–20% pre-estimate was wrong for bicubic —
recorded here so future estimates account for compiler CSE on pure duplicate calls.
Scalar (impl0) run-to-run noise is ±4–7%; trust the impl1 column.
