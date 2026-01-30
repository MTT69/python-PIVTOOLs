# Building Static GSL for Linux x86_64

## Problem

When building on Apple Silicon (ARM) Macs, Docker defaults to running ARM64 containers. This results in libraries compiled for `aarch64` instead of `x86_64`, which fail when used in x86_64 Linux environments (like manylinux wheels).

Error signature:
```
ld: libgsl.a(error.o): Relocations in generic ELF (EM: 183)
ld: libgsl.a: error adding symbols: file in wrong format
```

`EM: 183` indicates AArch64/ARM64 architecture.

## Solution

Use Docker's `--platform linux/amd64` flag to force x86_64 emulation.

## Build Command

```bash
docker run --platform linux/amd64 --rm \
  -v /path/to/static_gsl:/output \
  quay.io/pypa/manylinux2014_x86_64 \
  bash -c '
    set -e
    cd /tmp
    curl -L https://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz -o gsl-2.8.tar.gz
    tar xzf gsl-2.8.tar.gz
    cd gsl-2.8
    ./configure --prefix=/output/linux --enable-static --disable-shared CFLAGS="-O3 -fPIC"
    make -j$(nproc)
    make install
  '
```

## Key Points

- **`--platform linux/amd64`**: Forces x86_64 architecture regardless of host CPU
- **`manylinux2014_x86_64`**: Uses the official Python wheel build container for broad compatibility
- **`-fPIC`**: Position-independent code, required for shared library linking
- **`--enable-static --disable-shared`**: Build only static libraries

## Verification

Check the compiled architecture:
```bash
strings lib/libgsl.a | grep GCC
```

Should show x86_64 toolchain (e.g., `x86_64-redhat-linux`), not `aarch64-linux-gnu`.
