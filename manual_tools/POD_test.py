import numpy as np
import time
from threadpoolctl import threadpool_limits

H, W, N = 2048, 2048, 25
A = np.random.randn(H * W, N).astype(np.float32)

for threads in [1, 2, 4, 8, 10]:
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        print(f"{threads:2d} threads: {time.perf_counter() - start:.2f}s")