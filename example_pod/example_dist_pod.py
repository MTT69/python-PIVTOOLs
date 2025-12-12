import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dask import delayed
import dask.array as da
from dask.distributed import Client, LocalCluster
import imageio.v3 as iio
from pivtools_cli.preprocessing.filters import distributed_pod, _pod_filter_block

FOLDER = "/Users/connorourke/Scratch/HPC_RSE_PROJS/MorganTaylor/DATA/images_0_noise/Cam1"
MAX_PAIRS = 200        # << you can raise or lower this freely
BLOCK_SIZE = 10    

if __name__ == "__main__":
    assert os.path.isdir(FOLDER)

    cluster = LocalCluster(
        n_workers=5,
        threads_per_worker=1,
        memory_limit="10GB",      
        dashboard_address=":8788",
    )
    client = Client(cluster)
    #uses B_00000A.tif and B_00000B.tif format:
    files = [f for f in os.listdir(FOLDER) if f.lower().endswith(".tif")]

    def get_index(fname):
        digits = ''.join(ch for ch in fname if ch.isdigit())
        return int(digits) if digits else -1

    files.sort(key=get_index)

    pairs = {}
    for f in files:
        idx = get_index(f)
        if idx < 0:
            continue
        if "_A" in f.upper():
            pairs.setdefault(idx, {})["A"] = os.path.join(FOLDER, f)
        if "_B" in f.upper():
            pairs.setdefault(idx, {})["B"] = os.path.join(FOLDER, f)

    A_list = []
    B_list = []
    for idx in sorted(pairs.keys())[:MAX_PAIRS]:
        if "A" in pairs[idx] and "B" in pairs[idx]:
            A_list.append(pairs[idx]["A"])
            B_list.append(pairs[idx]["B"])

    N = len(A_list)
    logging.info(f"Using {N} pairs")

    sample = iio.imread(A_list[0])
    H, W = sample.shape
    img_shape = (H, W)
    del sample

    def lazy_read(path):
        @delayed
        def _read():
            return iio.imread(path).astype(np.float32)
        return da.from_delayed(_read(), shape=img_shape, dtype=np.float32)

    A_da = da.stack([lazy_read(p) for p in A_list])
    B_da = da.stack([lazy_read(p) for p in B_list])

    pairs_da = da.stack([A_da, B_da], axis=1) 

    pairs_da = pairs_da.rechunk((BLOCK_SIZE, 2, H, W))

    logging.info("pairs_da:", pairs_da.shape, pairs_da.chunks)


    filtered_dist = distributed_pod(
        pairs_da,
        eps_auto_psi=0.01,
        eps_auto_sigma=0.01,
    ).persist()       
    print("done distributed pod ")

    
    def run_original(pairs_dask):
        arr = pairs_dask.compute()  
        return _pod_filter_block(arr)
    fut = client.submit(run_original, pairs_da)
    filtered_single = fut.result()



    diff = filtered_dist.astype(np.float32) - filtered_single.astype(np.float32)
    absdiff = np.abs(diff)
    mae = absdiff.mean()
    maxerr = absdiff.max()

    logging.info(f"Mean absolute error = {mae}")
    logging.info(f"Max absolute error  = {maxerr}")

    i = 10
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0,0].imshow(filtered_single[i,0], cmap="gray")
    axes[0,0].set_title("Single-worker POD (A)")
    axes[0,1].imshow(filtered_dist[i,0], cmap="gray")
    axes[0,1].set_title("Distributed global POD (A)")
    im = axes[0,2].imshow(diff[i,0], cmap="bwr", vmin=-np.max(np.abs(diff[i,0])), vmax=np.max(np.abs(diff[i,0])))
    axes[0,2].set_title("Difference (A)")
    fig.colorbar(im, ax=axes[0,2])

    axes[1,0].imshow(filtered_single[i,1], cmap="gray")
    axes[1,0].set_title("Single-worker POD (B)")
    axes[1,1].imshow(filtered_dist[i,1], cmap="gray")
    axes[1,1].set_title("Distributed global POD (B)")
    im2 = axes[1,2].imshow(diff[i,1], cmap="bwr", vmin=-np.max(np.abs(diff[i,1])), vmax=np.max(np.abs(diff[i,1])))
    axes[1,2].set_title("Difference (B)")
    fig.colorbar(im2, ax=axes[1,2])

    plt.tight_layout()
    plt.savefig("comparison.png")