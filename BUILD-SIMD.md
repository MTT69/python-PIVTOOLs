# Building the SIMD codelet FFT (Stage B) per platform

`libbulkxcorr2d` runs an FFTW-free, self-owned codelet FFT on the PIV
cross-correlation hot path. Stage B processes **W PIV windows per SIMD register**
(one window per lane, zero cross-lane shuffles). Exactly one lane width is
chosen at compile time. This note is how to build + benchmark it on each box.

## Quick build

```
cd PyPIVTools           # the worktree you want to build
pip install -e .        # or: python setup.py build
```

The build prints what it resolved, e.g.:

```
>>> [PIVTOOLS_FFT] libbulkxcorr2d  isa=neon4  lanes=4  macro=PIVTOOLS_FFT_ISA_NEON4
    render=vecext4  arch='-mcpu=native'  lto=False
>>> [PIVTOOLS_FFT] bulk compile flags: -O3 -fPIC -fopenmp ... -DPIVTOOLS_FFT_ISA_NEON4 -mcpu=native ...
```

## Platform defaults

| Platform | Compiler | Default ISA | Lanes | Arch flag |
|---|---|---|---|---|
| macOS arm64 | Homebrew LLVM clang (`brew install llvm libomp`) | `neon4` | 4 (one NEON reg) | `-mcpu=native` |
| Linux x86_64 | `clang` (or `$CC`) | `vext8` | 8 (AVX2 via vector_size) | `-mavx2 -mfma` |
| Linux HPC (AVX-512) | `clang` | set `PIVTOOLS_FFT_ISA=avx512` | 16 (one zmm) | `-march=native` |
| Windows | clang-cl (VS LLVM tools) | `avx2` | 8 (AVX2 intrinsics) | `/arch:AVX2` |

The vector_size renders (neon4 / vext8 / avx512) are ISA-agnostic in source — the
**arch flag** (not the width) decides NEON vs AVX lowering, so on arm64 *every*
width uses `-mcpu=native`. Windows cannot use the `vector_size` extension under
the cl-compatible front end, so it takes the `avx2` intrinsic render (`__m256`).

## Env knobs (the porting / benchmarking levers)

| Variable | Effect |
|---|---|
| `PIVTOOLS_FFT_ISA` | Force the width: `neon4`, `vext8`, `avx512` (clang/gcc) or `avx2` (Windows). Use to A/B widths on one box. |
| `PIVTOOLS_FFT_MARCH` | Replace the arch flag verbatim, e.g. `-march=icelake-server`, `-march=skylake-avx512`. |
| `PIVTOOLS_FFT_LTO=1` | Enable link-time optimisation (off by default; fragile on some toolchains). |

Example — benchmark the 8-wide width on the Mac:
```
PIVTOOLS_FFT_ISA=vext8 pip install -e .
```

## HPC / Iridis caveat (important)

`-march=native` / `-mcpu=native` resolves to the **build host's** ISA. On a
cluster the login node often differs from the compute nodes — a login-node
`native` build may `SIGILL` on older compute nodes or under-use newer ones.
Either build inside an interactive job on a compute node, or pin the arch:

```
# AVX-512 on Iridis, pinned to the compute-node microarch:
PIVTOOLS_FFT_ISA=avx512 PIVTOOLS_FFT_MARCH='-march=icelake-server' pip install -e .
```

The build emits a NOTICE whenever a `native` arch flag is in play.

## Accuracy guardrails (do not change)

The build deliberately uses `-ffp-contract=fast` (value-safe FMA fusion, applied
consistently) and **never** `-ffast-math` / `/fp:fast` (those reassociate and
flush-to-zero, breaking the bit-for-bit FFTW parity the engine is validated
against). On the bailey dataset the SIMD output is **bit-identical to FFTW**.

## Verify the build

```
# FFTW-free (expect 0):
nm pivtools_cli/lib/libbulkxcorr2d.so | grep -ci fftw          # Linux/macOS
dumpbin /imports pivtools_cli\lib\libbulkxcorr2d.dll | findstr fftw   # Windows

# Confirm the compiled lane width:
python -c "import ctypes; l=ctypes.CDLL('pivtools_cli/lib/libbulkxcorr2d.so'); \
           l.codelet_lanes.restype=ctypes.c_int; print('lanes', l.codelet_lanes())"

# Standalone correctness gate (batched engine vs scalar oracle + brute force):
cd pivtools_cli/lib
"$(brew --prefix llvm)/bin/clang" -O3 -fopenmp -ffp-contract=fast -mcpu=native \
       -DPIVTOOLS_FFT_ISA_NEON4 \
       -I. codelet_fft.c test_codelet_gate.c -lm -o /tmp/gate && /tmp/gate   # -> GATE PASS
```

(Swap the `-D…` macro + arch flag to match the width you built:
`-DPIVTOOLS_FFT_ISA_VEXT8`, `-DPIVTOOLS_FFT_ISA_AVX512`, or on Windows the
`avx2` render with `/arch:AVX2`.)

## Notes

- `codelets_gen.h` is generated at build time (gitignored). clang/gcc builds
  emit the full vector_size family (scalar+v4+v8+v16) so the width can be
  switched via `PIVTOOLS_FFT_ISA` without regenerating; Windows emits
  scalar+avx2.
- Binary wheels (PyPI) are built portable, not `native`: x86 wheels pin
  AVX2+FMA (`PIVTOOLS_WARP_MARCH="-mavx2 -mfma"`) and macOS arm64 wheels pin
  `-mcpu=apple-m1`. The x86 floor is enforced at load time by
  `pivtools_cpu_supported()` (peak_locate_lm.c) — CPUs older than Haswell
  (2013) get a clear error telling them to install from sdist. The macOS wheel
  floor (`macosx_15_0`) is set by the Homebrew libomp bottle that delocate
  bundles, not by the toolchain.
- Linux wheels are `manylinux_2_28` (glibc >= 2.28, i.e. 2018+ distros:
  RHEL/Alma 8+, Ubuntu 20.04+, Debian 10+). This is forced by the toolchain —
  the older manylinux2014 image has no usable clang. On older distros (CentOS
  7, Ubuntu 18.04, dated HPC login nodes) pip falls back to the sdist, which
  compiles locally and needs clang + OpenMP.
- No library links FFTW or GSL any more: `libkspace` and `libmarquadt` were
  removed (2026-06-23) and ensemble fitting moved to pure NumPy, so the BSD-3
  wheel claim is now honest.
- Only **instantaneous** PIV is validated end-to-end so far; the batched
  ensemble/triple loops are built but pending joint validation.
