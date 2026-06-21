#!/usr/bin/env python3
"""Generate fully-unrolled, fixed-size FFT codelets for the PIV correlation hot path.

A codelet is a straight-line (no loops, no runtime trig) Cooley-Tukey FFT for one
fixed power-of-two N, with all twiddle factors baked in as constants and trivial
(+/-1, +/-i) twiddles folded to adds/negations. This triviality at small N is exactly
where FFTW's hand-tuned codelets beat generic libraries (pocketfft/PFFFT) -- see
`../../../Downloads/fftw research/HANDOFF.md`.

Design:
  - Build an IR (list of real-valued scalar ops) for each transform by recursing the
    radix-2 DIT butterfly network, with a tiny algebraic layer that folds 0/1/-1 and
    constant twiddles. rfft drops the (zero) imaginary input arithmetic for free.
  - Validate the IR against numpy BEFORE rendering any C (catch algorithm bugs early).
  - Render the SAME IR to two C bodies via a renderer object:
        scalar  -> plain `float`  (gate + serial backend, one window per call)
        v8      -> `__m256`        (8 windows per SIMD lane, the throughput backend)
    One window per lane => the butterfly network never shuffles across lanes; twiddles
    become broadcast constants. That is the win FFTW can't match on throughput.

Transforms emitted per N (FFTW unnormalized convention, canonical [N/2+1] real layout):
  rfft   : N real      -> N/2+1 complex   (row stage of 2D r2c)
  cfft   : N complex   -> N complex        (column stage, forward)
  icfft  : N complex   -> N complex        (column stage, inverse; +i twiddles, no norm)
  irfft  : N/2+1 cplx  -> N real           (row stage of 2D c2r; Hermitian resolve)

Usage:
    python gen_codelet.py --validate          # numpy check only, no file written
    python gen_codelet.py --emit codelets_gen.h   # write the C header
"""
from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# IR builder with a minimal algebraic-folding layer.
# A "term" is either ('c', float)  (a compile-time constant) or
#                    ('v', name)   (a previously-assigned scalar variable).
# ---------------------------------------------------------------------------
@dataclass
class Builder:
    ops: list = field(default_factory=list)      # (kind, dst, *args)
    _ctr: int = 0

    def _nv(self) -> str:
        self._ctr += 1
        return f"t{self._ctr}"

    # real ops, with folding -------------------------------------------------
    def radd(self, a, b):
        if a[0] == 'c' and b[0] == 'c':
            return ('c', a[1] + b[1])
        if a[0] == 'c' and a[1] == 0.0:
            return b
        if b[0] == 'c' and b[1] == 0.0:
            return a
        v = self._nv()
        self.ops.append(('add', v, a, b))
        return ('v', v)

    def rsub(self, a, b):
        if a[0] == 'c' and b[0] == 'c':
            return ('c', a[1] - b[1])
        if b[0] == 'c' and b[1] == 0.0:
            return a
        if a[0] == 'c' and a[1] == 0.0:
            return self.rneg(b)
        v = self._nv()
        self.ops.append(('sub', v, a, b))
        return ('v', v)

    def rneg(self, a):
        if a[0] == 'c':
            return ('c', -a[1])
        v = self._nv()
        self.ops.append(('neg', v, a))
        return ('v', v)

    def rmulc(self, a, c: float):
        c = float(c)
        if c == 0.0:
            return ('c', 0.0)
        if c == 1.0:
            return a
        if c == -1.0:
            return self.rneg(a)
        if a[0] == 'c':
            return ('c', a[1] * c)
        v = self._nv()
        self.ops.append(('mulc', v, a, c))
        return ('v', v)

    # complex helpers (each complex value is a (re_term, im_term) pair) ------
    def cadd(self, a, b):
        return (self.radd(a[0], b[0]), self.radd(a[1], b[1]))

    def csub(self, a, b):
        return (self.rsub(a[0], b[0]), self.rsub(a[1], b[1]))

    def cmul_const(self, a, w: complex):
        """Multiply complex value `a` by compile-time constant twiddle `w`."""
        wr, wi = w.real, w.imag
        # snap tiny values so +/-1, +/-i, 0 fold cleanly
        wr = 0.0 if abs(wr) < 1e-15 else wr
        wi = 0.0 if abs(wi) < 1e-15 else wi
        re = self.rsub(self.rmulc(a[0], wr), self.rmulc(a[1], wi))
        im = self.radd(self.rmulc(a[0], wi), self.rmulc(a[1], wr))
        return (re, im)


def _smallest_prime_factor(n: int) -> int:
    f = 2
    while f * f <= n:
        if n % f == 0:
            return f
        f += 1
    return n


def _fft(bld: Builder, xs: list, inverse: bool) -> list:
    """Recursive mixed-radix DIT on complex value-handles `xs`. Returns N handles.

    Splits by the smallest prime factor r (2 for power-of-two sizes, 3 for the
    2^k*3 sizes 12/24/48/96), recurses on the r decimated subsequences, then
    combines with a radix-r butterfly:
        T_j   = U_j[s] * W_N^{j*s}                 (twiddle each sub-output once)
        X[s+p*m] = sum_j T_j * W_r^{j*p}           (r-point DFT of the T_j)
    For r=2 this reproduces the exact +/-1 butterfly the original radix-2 path used,
    so the power-of-two codelets are unchanged (their smallest factor is always 2)."""
    n = len(xs)
    if n == 1:
        return [xs[0]]
    r = _smallest_prime_factor(n)
    m = n // r
    subs = [_fft(bld, xs[j::r], inverse) for j in range(r)]
    sign = 1.0 if inverse else -1.0
    out = [None] * n
    for s in range(m):
        T = [bld.cmul_const(subs[j][s], cmath.exp(sign * 2j * math.pi * (j * s) / n))
             for j in range(r)]
        for p in range(r):
            acc = None
            for j in range(r):
                term = bld.cmul_const(T[j], cmath.exp(sign * 2j * math.pi * (j * p) / r))
                acc = term if acc is None else bld.cadd(acc, term)
            out[s + p * m] = acc
    return out


# ---------------------------------------------------------------------------
# Build one transform: returns (ops, inputs, outputs)
#   inputs : ordered list of input scalar variable names the function reads
#   outputs: list of (kind, idx, term) -- kind in {'re','im','out'}
# ---------------------------------------------------------------------------
def build_cfft(n: int, inverse: bool):
    bld = Builder()
    xs = [(('v', f"xr{i}"), ('v', f"xi{i}")) for i in range(n)]
    ys = _fft(bld, xs, inverse)
    inputs = [f"xr{i}" for i in range(n)] + [f"xi{i}" for i in range(n)]
    outputs = []
    for k in range(n):
        outputs.append(('re', k, ys[k][0]))
        outputs.append(('im', k, ys[k][1]))
    return prune(bld.ops, outputs), inputs, outputs


def build_rfft(n: int):
    """Real input (imag = 0); store only the canonical N/2+1 outputs."""
    bld = Builder()
    xs = [(('v', f"xr{i}"), ('c', 0.0)) for i in range(n)]
    ys = _fft(bld, xs, inverse=False)
    inputs = [f"xr{i}" for i in range(n)]
    outputs = []
    for k in range(n // 2 + 1):
        outputs.append(('re', k, ys[k][0]))
        outputs.append(('im', k, ys[k][1]))
    return prune(bld.ops, outputs), inputs, outputs


def build_irfft(n: int):
    """Hermitian input X[0..N/2] -> N real outputs (FFTW c2r, unnormalized).

    Full spectrum rebuilt by symmetry X[N-k] = conj(X[k]); inverse complex FFT;
    keep the real part (imag is ~0 by construction)."""
    bld = Builder()
    half = n // 2
    xr = {k: ('v', f"Xr{k}") for k in range(half + 1)}
    xi = {k: ('v', f"Xi{k}") for k in range(half + 1)}
    # DC (k=0) and Nyquist (k=N/2) imag parts are structurally zero in a real signal.
    xi[0] = ('c', 0.0)
    xi[half] = ('c', 0.0)
    xs = []
    for k in range(n):
        if k <= half:
            xs.append((xr[k], xi[k]))
        else:                                   # conj symmetry
            xs.append((xr[n - k], bld.rneg(xi[n - k])))
    ys = _fft(bld, xs, inverse=True)
    inputs = [f"Xr{k}" for k in range(half + 1)] + [f"Xi{k}" for k in range(1, half)]
    outputs = [('out', k, ys[k][0]) for k in range(n)]   # real part only
    return prune(bld.ops, outputs), inputs, outputs


# ---------------------------------------------------------------------------
# Dead-code elimination: keep only ops reachable from stored outputs.
# ---------------------------------------------------------------------------
def prune(ops: list, outputs: list) -> list:
    by_dst = {op[1]: op for op in ops}
    needed = set()
    stack = []
    for _, _, term in outputs:
        if term[0] == 'v':
            stack.append(term[1])
    while stack:
        v = stack.pop()
        if v in needed or v not in by_dst:
            continue
        needed.add(v)
        op = by_dst[v]
        for arg in op[2:]:
            if isinstance(arg, tuple) and arg[0] == 'v':
                stack.append(arg[1])
    return [op for op in ops if op[1] in needed]


# ---------------------------------------------------------------------------
# Validation: interpret the IR with numpy and compare to numpy.fft.
# ---------------------------------------------------------------------------
def interpret(ops, outputs, env: dict) -> dict:
    def val(term):
        return term[1] if term[0] == 'c' else env[term[1]]
    for op in ops:
        k = op[0]
        if k == 'add':
            env[op[1]] = val(op[2]) + val(op[3])
        elif k == 'sub':
            env[op[1]] = val(op[2]) - val(op[3])
        elif k == 'neg':
            env[op[1]] = -val(op[2])
        elif k == 'mulc':
            env[op[1]] = val(op[2]) * op[3]
        else:
            raise ValueError(k)
    res = {}
    for kind, idx, term in outputs:
        res[(kind, idx)] = val(term)
    return res


def validate(sizes):
    import numpy as np
    rng = np.random.default_rng(0)
    ok = True
    for n in sizes:
        # cfft forward
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ops, _, outs = build_cfft(n, inverse=False)
        env = {}
        for i in range(n):
            env[f"xr{i}"] = x[i].real
            env[f"xi{i}"] = x[i].imag
        r = interpret(ops, outs, env)
        y = np.array([r[('re', k)] + 1j * r[('im', k)] for k in range(n)])
        e1 = np.max(np.abs(y - np.fft.fft(x)))
        # icfft (inverse, unnormalized => N * ifft)
        ops, _, outs = build_cfft(n, inverse=True)
        env = {}
        for i in range(n):
            env[f"xr{i}"] = x[i].real
            env[f"xi{i}"] = x[i].imag
        r = interpret(ops, outs, env)
        y = np.array([r[('re', k)] + 1j * r[('im', k)] for k in range(n)])
        e2 = np.max(np.abs(y - n * np.fft.ifft(x)))
        # rfft
        xr = rng.standard_normal(n)
        ops, _, outs = build_rfft(n)
        env = {f"xr{i}": xr[i] for i in range(n)}
        r = interpret(ops, outs, env)
        y = np.array([r[('re', k)] + 1j * r[('im', k)] for k in range(n // 2 + 1)])
        e3 = np.max(np.abs(y - np.fft.rfft(xr)))
        # irfft (c2r): irfft(rfft(x)) == N*x
        X = np.fft.rfft(xr)
        ops, _, outs = build_irfft(n)
        env = {}
        for k in range(n // 2 + 1):
            env[f"Xr{k}"] = X[k].real
        for k in range(1, n // 2):
            env[f"Xi{k}"] = X[k].imag
        r = interpret(ops, outs, env)
        y = np.array([r[('out', k)] for k in range(n)])
        e4 = np.max(np.abs(y - n * xr))
        tag = "OK" if max(e1, e2, e3, e4) < 1e-9 else "**FAIL**"
        if tag != "OK":
            ok = False
        print(f"N={n:>4}  cfft={e1:.2e}  icfft={e2:.2e}  rfft={e3:.2e}  irfft={e4:.2e}  {tag}")
    return ok


# ---------------------------------------------------------------------------
# C rendering.  One IR -> scalar `float` or SIMD (__m256 v8 / __m512 v16) C,
# selected by an ISA descriptor.  The lane width is a pure renderer parameter:
# the AVX-512 (Iridis) codelet is the SAME IR rendered with --isa avx512.
# ---------------------------------------------------------------------------
@dataclass
class Isa:
    name: str          # 'scalar' | 'avx2' | 'avx512' | 'vecext'
    ctype: str         # C element type
    lanes: int         # windows processed per call (1 for scalar)
    suffix: str        # function-name suffix
    simd_ops: bool = False   # True -> intrinsic macros (avx*); False -> C operators
                             # (scalar float AND GCC/Clang vector_size both use operators)

    def lit(self, c: float) -> str:
        s = repr(float(c))
        if "e" not in s and "." not in s:
            s += ".0"
        s += "f"
        return f"set1({s})" if self.simd_ops else s

    def add(self, a, b): return f"vadd({a}, {b})" if self.simd_ops else f"({a} + {b})"
    def sub(self, a, b): return f"vsub({a}, {b})" if self.simd_ops else f"({a} - {b})"
    def neg(self, a):
        return f"vsub(set1(0.0f), {a})" if self.simd_ops else f"(-{a})"
    def mulc(self, a, c):
        return f"vmul({a}, {self.lit(c)})" if self.simd_ops else f"({a} * {self.lit(c)})"

    def bcast_const(self, c) -> str:
        """A constant assigned *directly* to an output lane (e.g. the structurally-zero
        imaginary DC/Nyquist bins) must be broadcast to every lane on vector ISAs.
        Intrinsic ISAs already broadcast via set1() inside lit(); the GCC/Clang
        vector_size path needs an explicit per-lane init (scalar->vector assignment is
        rejected, unlike scalar->vector *arithmetic* which broadcasts fine)."""
        if self.simd_ops or self.lanes == 1:
            return self.lit(c)                       # set1(...) for avx*, bare literal for scalar
        s = self.lit(c)
        return f"(({self.ctype}){{{', '.join([s] * self.lanes)}}})"


ISAS = {
    "scalar": Isa("scalar", "float", 1, "scalar", simd_ops=False),
    "avx2":   Isa("avx2", "__m256", 8, "v8", simd_ops=True),
    "avx512": Isa("avx512", "__m512", 16, "v16", simd_ops=True),
    # GCC/Clang vector extensions: `a+b`, `a*c` (scalar broadcast) just work and the
    # compiler lowers to AVX2 on x86 or NEON (2 regs) on arm64. One source, both Unix
    # platforms. ctype `v8` is typedef'd in the header. NOT supported by MSVC.
    "vecext": Isa("vecext", "v8", 8, "v8", simd_ops=False),
    # Native-width NEON / SSE batch: vector_size(16) = 4 floats = ONE NEON (or SSE) register,
    # so no 2-register split. Same operator lowering as vecext. ctype `v4`. NOT MSVC.
    "vecext4": Isa("vecext4", "v4", 4, "v4", simd_ops=False),
    # AVX-512 native batch for Linux/GCC (Iridis): vector_size(64) = 16 floats = ONE zmm
    # register when built `-march=native`. The intrinsic `avx512` ISA above is the MSVC
    # equivalent; this one needs no intrinsics. ctype `v16`. NOT MSVC.
    "vecext16": Isa("vecext16", "v16", 16, "v16", simd_ops=False),
}


def _render_fn(isa: Isa, ret_sig, ops, outputs) -> str:
    """Emit one function body. `ret_sig` already contains the signature head."""
    lines = [ret_sig + " {"]
    for op in ops:
        def a(t):
            return isa.lit(t[1]) if t[0] == 'c' else t[1]
        k, dst = op[0], op[1]
        if k == 'add':
            rhs = isa.add(a(op[2]), a(op[3]))
        elif k == 'sub':
            rhs = isa.sub(a(op[2]), a(op[3]))
        elif k == 'neg':
            rhs = isa.neg(a(op[2]))
        elif k == 'mulc':
            rhs = isa.mulc(a(op[2]), op[3])
        else:
            raise ValueError(k)
        lines.append(f"    const {isa.ctype} {dst} = {rhs};")
    for kind, idx, term in outputs:
        v = isa.bcast_const(term[1]) if term[0] == 'c' else term[1]
        if kind == 're':
            lines.append(f"    Yr[{idx}] = {v};")
        elif kind == 'im':
            lines.append(f"    Yi[{idx}] = {v};")
        else:
            lines.append(f"    y[{idx}] = {v};")
    lines.append("}")
    return "\n".join(lines)


def render_header(sizes, isas) -> str:
    out = ["// AUTO-GENERATED by gen_codelet.py -- do not edit.",
           "// Fixed-size unrolled FFT codelets (FFTW unnormalized convention,",
           "// canonical [N/2+1] real layout). Same IR rendered per ISA; the lane",
           "// width is a renderer parameter, so AVX-512 (Iridis) is --isa avx512.",
           "#pragma once", ""]
    # Preamble depends on which SIMD ISA (at most one) is present.
    intrin = [i for i in isas if ISAS[i].simd_ops]      # avx2 / avx512 (intrinsics)
    # vector_size family (vecext=8-wide, vecext4=4-wide): operator-based, no macros, so
    # several can coexist in one header — each just needs its own typedef.
    vecext = [i for i in isas if not ISAS[i].simd_ops and ISAS[i].lanes > 1]
    if intrin:
        isa = ISAS[intrin[0]]
        w = "256" if isa.name == "avx2" else "512"
        out += ["#include <immintrin.h>", "",
                f"// --- {isa.name} vector ops ---",
                f"#define vadd(a,b) _mm{w}_add_ps((a),(b))",
                f"#define vsub(a,b) _mm{w}_sub_ps((a),(b))",
                f"#define vmul(a,b) _mm{w}_mul_ps((a),(b))",
                f"#define set1(x)   _mm{w}_set1_ps((x))", ""]
    elif vecext:
        out += [f"// --- GCC/Clang vector extensions (lower to AVX2/SSE on x86 / NEON on arm64) ---"]
        for iname in vecext:
            isa = ISAS[iname]
            out.append(f"typedef float {isa.ctype} __attribute__((vector_size({isa.lanes * 4})));")
        out.append("")
    for n in sizes:
        half = n // 2
        for iname in isas:
            isa = ISAS[iname]
            t, s = isa.ctype, isa.suffix
            # rfft (real input -> N/2+1 complex)
            ops, _, outs = build_rfft(n)
            sig = f"static inline void rfft{n}_{s}(const {t}* xr, {t}* Yr, {t}* Yi)"
            out.append(_emit(isa, sig, ops, outs,
                             {f"xr{i}": f"xr[{i}]" for i in range(n)}))
            # cfft (forward) and icfft (inverse)
            for inv, fn in ((False, "cfft"), (True, "icfft")):
                ops, _, outs = build_cfft(n, inverse=inv)
                sig = (f"static inline void {fn}{n}_{s}"
                       f"(const {t}* Xr, const {t}* Xi, {t}* Yr, {t}* Yi)")
                binds = {**{f"xr{i}": f"Xr[{i}]" for i in range(n)},
                         **{f"xi{i}": f"Xi[{i}]" for i in range(n)}}
                out.append(_emit(isa, sig, ops, outs, binds))
            # irfft (c2r)
            ops, _, outs = build_irfft(n)
            sig = f"static inline void irfft{n}_{s}(const {t}* Xr, const {t}* Xi, {t}* y)"
            binds = {**{f"Xr{k}": f"Xr[{k}]" for k in range(half + 1)},
                     **{f"Xi{k}": f"Xi[{k}]" for k in range(1, half)}}
            out.append(_emit(isa, sig, ops, outs, binds))
    return "\n\n".join(out) + "\n"


def _emit(isa: Isa, sig: str, ops, outputs, binds) -> str:
    """Render a function: bind named inputs to array reads, then the IR body."""
    head = [sig + " {"]
    for name, expr in binds.items():
        head.append(f"    const {isa.ctype} {name} = {expr};")
    body = _render_fn(isa, "", ops, outputs)
    # strip the placeholder head ("" + " {") that _render_fn added
    body_lines = body.split("\n")[1:-1]   # drop ' {' and '}'
    return "\n".join(head + body_lines + ["}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--emit", type=str, default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--isa", nargs="+", default=["scalar", "avx2"],
                    choices=list(ISAS.keys()),
                    help="ISAs to emit. At most one SIMD ISA per header "
                         "(avx2 OR avx512). Iridis: --isa scalar avx512.")
    args = ap.parse_args()
    if args.validate or not args.emit:
        ok = validate(args.sizes)
        print("ALL PASS" if ok else "SOME FAILED")
        if not args.emit:
            return
    # Only intrinsic ISAs (avx2/avx512) share the vadd/vmul/set1 macro names, so at most
    # one of those per header. vector_size ISAs (vecext/vecext4) use distinct typedefs and
    # plain operators, so any number can coexist.
    intrin = [i for i in args.isa if ISAS[i].simd_ops]
    if len(intrin) > 1:
        raise SystemExit("emit at most one intrinsic SIMD ISA per header (macros collide)")
    hdr = render_header(args.sizes, args.isa)
    with open(args.emit, "w") as f:
        f.write(hdr)
    print(f"wrote {args.emit}  ({len(hdr.splitlines())} lines, "
          f"sizes={args.sizes}, isa={args.isa})")


if __name__ == "__main__":
    main()
