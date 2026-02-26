Here is Claude's plan:                                                                                          
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Integrate Fused Warp C Kernel into Production

 Context

 The standalone fused_symmetric_warp C kernel is complete and tested (pivtools_cli/lib/fused_warp.c). It fuses
 predictor upsampling + symmetric coordinate maps + bicubic/Lanczos-3 image warping into a single OpenMP pass.

 Measured speedups vs the current cv2.remap production pipeline (single-threaded, best of 5):

 ┌──────────┬─────────────┬─────────────┬─────────────┬─────────────┬──────────────┬──────────────┐
 │ Size     │ cv2 cubic   │ cv2 lanc4   │ C bicubic   │ C lanczos3  │ Speedup      │ Speedup      │
 │          │ (prod now)  │ (prod alt)  │ (fused)     │ (fused+LUT) │ cub C/cv2    │ lan C/cv2    │
 ├──────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────────┼──────────────┤
 │ 1 MP     │ 17.3 ms     │ 19.1 ms     │ 5.8 ms      │ 9.3 ms      │ 3.0x         │ 2.0x         │
 │ 4 MP     │ 64.1 ms     │ 67.4 ms     │ 9.6 ms      │ 14.4 ms     │ 6.7x         │ 4.7x         │
 │ 25 MP    │ 355.3 ms    │ 351.0 ms    │ 61.5 ms     │ 83.7 ms     │ 5.8x         │ 4.2x         │
 └──────────┴─────────────┴─────────────┴─────────────┴─────────────┴──────────────┴──────────────┘

 C Lanczos-3 vs C bicubic ratio: ~1.5x (from the larger 6×6 stencil — irreducible memory cost).
 The LUT eliminated the sinf bottleneck that initially made Lanczos 5-6x slower than bicubic.

 This plan integrates the kernel into the production cpu_instantaneous.py and cpu_ensemble.py pipelines and adds 
  it to the build system. The fused kernel replaces the cv2.remap pipeline entirely (no fallback) — libfusedwarp 
  is a hard requirement like libbulkxcorr2d.

 ---
 Files to Modify

 ┌─────┬───────────────────────────────────────────────────┬─────────────────────────────────────────────────┐   
 │  #  │                       File                        │                     Change                      │   
 ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────────────────┤   
 │ 1   │ setup.py                                          │ Add libfusedwarp build step (no FFTW/GSL deps)  │   
 ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────────────────┤   
 │ 2   │ pyproject.toml                                    │ Add source + binary to package-data             │   
 ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────────────────┤   
 │ 3   │ pivtools_cli/piv/piv_backend/cpu_instantaneous.py │ Load lib, replace dense remap + mesh + image    │   
 │     │                                                   │ warp with fused call                            │   
 ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────────────────┤   
 │ 4   │ pivtools_cli/piv/piv_backend/cpu_ensemble.py      │ Load lib, replace dense remap + mesh + image    │   
 │     │                                                   │ warp with fused call                            │   
 ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────────────────┤   
 │ 5   │ pivtools_gui/app.py                               │ Add libfusedwarp to system_info c_libraries     │   
 │     │                                                   │ check                                           │   
 ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────────────────┤   
 │ 6   │ CLAUDE.md                                         │ Document the new library                        │   
 └─────┴───────────────────────────────────────────────────┴─────────────────────────────────────────────────┘   

 No new files needed — fused_warp.h and fused_warp.c already exist but will be extended with a batch function.   

 ---
 Step 1: Build System — setup.py (lines 235-238)

 Add after the libinterp2custom build block (line 235), before _build_marquadt():

 # --- Build libfusedwarp (no FFTW/GSL) ---
 if use_msvc:
     output_file = build_dir / f"libfusedwarp{lib_ext}"
     cmd_fw = [
         compiler, *self.extra_compile, shared_flag,
         f"/Fo{build_dir}/",
         str(src_dir / "fused_warp.c"),
         f"/I{src_dir}",
         f"/Fe{output_file}"
     ]
 else:
     cmd_fw = [
         compiler, *self.extra_compile, shared_flag,
         str(src_dir / "fused_warp.c"),
         f"-I{src_dir}",
         "-o", str(build_dir / f"libfusedwarp{lib_ext}"),
         "-lm", "-fopenmp"
     ]
 self._run(cmd_fw)
 if not (build_dir / f"libfusedwarp{lib_ext}").exists():
     raise RuntimeError(f"Build failed: libfusedwarp{lib_ext} not created")
 self._cleanup_intermediates(build_dir)

 Key: No FFTW or GSL link flags — only needs OpenMP + math.

 ---
 Step 2: Package Metadata — pyproject.toml (line 93)

 Add to [tool.setuptools.package-data] "pivtools_cli" list, after libmarquadt.*:

 "lib/fused_warp.c",
 "lib/fused_warp.h",
 "lib/libfusedwarp.*",

 ---
 Step 3: Batch C Function — fused_warp.h + fused_warp.c

 Add fused_symmetric_warp_batch alongside the existing single-pair function. Sends all N image pairs to C at     
 once, loops internally — avoids N Python→C round-trips and enables better OpenMP parallelism.

 3a. Header addition (fused_warp.h)

 /*
  * Batch version: warp N image pairs.
  *
  * Images are stacked as (N, H, W) in row-major order.
  * Predictor can be shared (ensemble) or per-image (instantaneous):
  *   shared_predictor=1: pred_dy/dx are (nPY, nPX) — same for all images
  *   shared_predictor=0: pred_dy/dx are (N, nPY, nPX) — separate per image
  *
  * OpenMP parallelizes over (image, row) with collapse(2).
  */
 EXPORT int fused_symmetric_warp_batch(
     const float *imgs_a,       /* (N, H, W) stacked */
     const float *imgs_b,       /* (N, H, W) stacked */
     float       *outs_a,       /* (N, H, W) stacked */
     float       *outs_b,       /* (N, H, W) stacked */
     const float *pred_dy,      /* (nPY, nPX) if shared, (N, nPY, nPX) if per-image */
     const float *pred_dx,      /* same */
     int N,
     int H, int W,
     int nPY, int nPX,
     const float *ctrs_y,
     const float *ctrs_x,
     int interp_mode,
     int shared_predictor       /* 1=shared (ensemble), 0=per-image (instantaneous) */
 );

 3b. Implementation (fused_warp.c)

 Also add Lanczos-3 interpolation as interp_mode=1 for image warping (predictor upsampling stays bicubic
 regardless — smooth displacement field doesn't benefit from Lanczos).

 3c. Lanczos-3 kernel (6×6 stencil) — LUT-accelerated

 The naive implementation uses sinf() calls (12 per pixel: 6 y-weights + 6 x-weights), which dominates runtime
 and makes Lanczos 5-6x slower than bicubic. Solution: a precomputed weight LUT (4096 entries × 6 weights ≈ 96 KB)
 with linear interpolation between entries. Built once before the OpenMP parallel region, read-only shared across
 threads.

 #define LANCZOS3_LUT_SIZE 4096

 /* Exact weight (used only to build the LUT) */
 static inline float lanczos_weight(float t, int a) {
     float at = fabsf(t);
     if (at < 1e-6f) return 1.0f;
     if (at >= (float)a) return 0.0f;
     float pi_t = (float)M_PI * at;
     float pi_t_a = pi_t / (float)a;
     return (sinf(pi_t) / pi_t) * (sinf(pi_t_a) / pi_t_a);
 }

 /* Build LUT: lut[i][k] = weight for fractional offset i/LUT_SIZE, stencil tap k */
 static void build_lanczos3_lut(float (*lut)[6], int size) {
     for (int i = 0; i <= size; i++) {
         float d = (float)i / (float)size;
         for (int k = 0; k < 6; k++)
             lut[i][k] = lanczos_weight(d - (float)(k - 2), 3);
     }
 }

 /* Fast weight lookup with linear interpolation */
 static inline void lanczos3_weights_6_lut(float d, const float (*lut)[6], float w[6]) {
     float pos = d * (float)LANCZOS3_LUT_SIZE;
     int idx = (int)pos;
     float frac = pos - (float)idx;
     if (idx >= LANCZOS3_LUT_SIZE) { idx = LANCZOS3_LUT_SIZE - 1; frac = 0.0f; }
     for (int k = 0; k < 6; k++)
         w[k] = lut[idx][k] + frac * (lut[idx + 1][k] - lut[idx][k]);
 }

 /* Lanczos-3 sample: 6×6 stencil, LUT weights, BORDER_CONSTANT=0 */
 static inline float lanczos3_sample(const float *img, float fy, float fx,
                                     int H, int W, const float (*lut)[6]) {
     float fy_floor = floorf(fy);
     float fx_floor = floorf(fx);
     int iy = (int)fy_floor - 2;  /* stencil starts at floor-2 */
     int ix = (int)fx_floor - 2;
     float dy = fy - fy_floor;
     float dx = fx - fx_floor;
     float wy[6], wx[6];
     float val = 0.0f;

     lanczos3_weights_6_lut(dy, lut, wy);
     lanczos3_weights_6_lut(dx, lut, wx);

     for (int m = 0; m < 6; m++) {
         int row = iy + m;
         if (row < 0 || row >= H) continue;
         for (int n = 0; n < 6; n++) {
             int col = ix + n;
             if (col < 0 || col >= W) continue;
             val += wy[m] * wx[n] * img[row * W + col];
         }
     }
     return val;
 }

 interp_mode mapping:
 - 0 = bicubic (Keys a=-0.75, 4×4 stencil) — matches cv2.INTER_CUBIC
 - 1 = Lanczos-3 (6×6 stencil) — matches cv2.INTER_LANCZOS4 quality class

 Performance: With LUT, Lanczos-3 is ~1.5x slower than bicubic (irreducible — 36 vs 16 stencil taps).
 Without LUT, sinf dominates at 5-6x. The LUT (96 KB) fits comfortably in L2 cache.

 3d. Batch function implementation

 Uses the same helpers (keys_weights_4, bicubic_sample, lanczos3_sample, bicubic_pred_wy,       
 build_pred_index_lut). The batch function:

 1. Builds 1D LUTs once (shared across all images — same ctrs_y, ctrs_x)
 2. Outer OpenMP loop: collapse(2) over (n, i) where n=0..N-1, i=0..H-1
 3. Per iteration: compute predictor pointer as pred_dy + (shared ? 0 : n * nPY * nPX), then same Phase A/B/C    
 logic as single-pair

 EXPORT int fused_symmetric_warp_batch(
     const float *imgs_a, const float *imgs_b,
     float *outs_a, float *outs_b,
     const float *pred_dy, const float *pred_dx,
     int N, int H, int W,
     int nPY, int nPX,
     const float *ctrs_y, const float *ctrs_x,
     int interp_mode, int shared_predictor
 ) {
     /* Input validation */
     if (N <= 0 || H <= 0 || W <= 0 || nPY <= 0 || nPX <= 0) return ERROR_NOMEM;
     /* ... null checks ... */

     /* Build shared 1D LUTs (once for all images) */
     float *pred_idx_y = malloc(H * sizeof(float));
     float *pred_idx_x = malloc(W * sizeof(float));
     /* ... build_pred_index_lut ... */

     int pred_stride = shared_predictor ? 0 : nPY * nPX;

     if (interp_mode == 0) {
         /* Bicubic */
         int ni;
         #pragma omp parallel for schedule(static) collapse(2)
         for (ni = 0; ni < N; ni++) {
             for (int i = 0; i < H; i++) {
                 const float *cur_pred_dy = pred_dy + ni * pred_stride;
                 const float *cur_pred_dx = pred_dx + ni * pred_stride;
                 const float *cur_img_a = imgs_a + ni * H * W;
                 const float *cur_img_b = imgs_b + ni * H * W;
                 float *cur_out_a = outs_a + ni * H * W;
                 float *cur_out_b = outs_b + ni * H * W;
                 /* ... same row logic as single-pair ... */
             }
         }
     } else {
         /* Lanczos-3 — build LUT once, shared read-only across OMP threads */
         float (*lanc_lut)[6] = malloc((LANCZOS3_LUT_SIZE + 1) * 6 * sizeof(float));
         if (!lanc_lut) { free(pred_idx_y); free(pred_idx_x); return ERROR_NOMEM; }
         build_lanczos3_lut(lanc_lut, LANCZOS3_LUT_SIZE);

         #pragma omp parallel for schedule(static) collapse(2)
         for (ni = 0; ni < N; ni++) {
             for (int i = 0; i < H; i++) {
                 /* ... same Phase A/B as bicubic, then: */
                 /* Phase C: lanczos3_sample(..., lanc_lut) instead of bicubic_sample */
             }
         }
         free(lanc_lut);
     }

     free(pred_idx_y); free(pred_idx_x);
     return ERROR_NONE;
 }

 Why batch? Same principle as bulkxcorr2d — send all work to C in one call, parallelize internally:
 - Eliminates N Python→C transitions — each ctypes call has overhead (~10µs), noticeable at high N
 - Better OpenMP work distribution — collapse(2) over N×H rows (e.g., 12 × 2048 = 24576 units) gives much finer  
 granularity than N separate calls of 2048 rows each. No thread spin-up/down between calls.
 - 1D LUTs built once — pred_idx_y[H] and pred_idx_x[W] are shared across all N images, computed once in the     
 batch function
 - Visible speedup — combined with eliminating delta_ab_dense allocation, cv2.remap overhead, and Python mesh    
 arithmetic, this should give a measurable improvement on the predictor-corrector step

 ---
 Step 4: Instantaneous Integration — cpu_instantaneous.py

 4a. Load library (in __init__, after libbulkxcorr2d load, ~line 47)

 Follow the existing libbulkxcorr2d pattern — hard requirement, FileNotFoundError if missing:

 # Load fused warp library (required)
 fw_path = os.path.join(
     os.path.dirname(__file__), "../..", "lib", f"libfusedwarp{lib_extension}"
 )
 fw_path = os.path.abspath(fw_path)
 if not os.path.isfile(fw_path):
     raise FileNotFoundError(f"Required library file not found: {fw_path}")
 self._fw_lib = ctypes.CDLL(fw_path)
 self._fw_lib.fused_symmetric_warp_batch.restype = ctypes.c_int
 self._fw_lib.fused_symmetric_warp_batch.argtypes = [
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_a (N,H,W)
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # imgs_b (N,H,W)
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_a (N,H,W)
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # outs_b (N,H,W)
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dy
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # pred_dx
     ctypes.c_int,                # N
     ctypes.c_int, ctypes.c_int,  # H, W
     ctypes.c_int, ctypes.c_int,  # nPY, nPX
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_y
     np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # ctrs_x
     ctypes.c_int,                # interp_mode
     ctypes.c_int,                # shared_predictor
 ]

 4b. Replace warp path in _predictor_corrector_batch (lines 700-760)

 What the fused kernel replaces:
 - pc_dense_and_predictor_remap section: dense cv2.remap (lines 713-718) — replaced
 - pc_mesh_and_image_warp section: mesh construction + image warp (lines 742-757) — replaced

 What stays in Python:
 - Predictor-to-window remap (_remap_predictor, lines 720-725) — small grid, negligible cost, needed for
 delta_ab_pred output
 - Gaussian smoothing (before this block)

 New code — single batch call to C with shared_predictor=0 (per-image predictors).

 Arrays are already float32 C-contiguous from their creation (numpy defaults). No ascontiguousarray wrapping     
 needed.

 with self._profile_section(pass_idx, "pc_predictor_remap"):
     # Predictor-to-window remap stays in Python (small grid, needed for output)
     map_x, map_y = self.cached_predictor_maps[pass_idx]
     def _remap_predictor(i, d):
         self.delta_ab_pred[i, ..., d] = cv2.remap(
             self.delta_ab_old[i, ..., d],
             map_x, map_y, interp_flag,
             borderMode=cv2.BORDER_REPLICATE,
         ).astype(np.float32)
     self._run_parallel(_remap_predictor,
                       [(i, d) for i in range(N) for d in range(2)])

 with self._profile_section(pass_idx, "pc_fused_warp"):
     ctrs_y = self.win_ctrs_y_all[pass_idx - 1]  # already float32 from cache
     ctrs_x = self.win_ctrs_x_all[pass_idx - 1]
     interp_mode = 0 if interpolator == "cubic" else 1  # 0=bicubic, 1=lanczos
     nPY, nPX = self.delta_ab_old.shape[1], self.delta_ab_old.shape[2]

     image_a_prime_batch = np.zeros((N, H, W), dtype=np.float32)
     image_b_prime_batch = np.zeros((N, H, W), dtype=np.float32)

     ret = self._fw_lib.fused_symmetric_warp_batch(
         images_a, images_b,                  # (N, H, W) inputs
         image_a_prime_batch, image_b_prime_batch,  # (N, H, W) outputs
         self.delta_ab_old[..., 0],           # pred_dy (N, nPY, nPX)
         self.delta_ab_old[..., 1],           # pred_dx (N, nPY, nPX)
         N, H, W, nPY, nPX,
         ctrs_y, ctrs_x,
         interp_mode,
         0,  # shared_predictor=0 → per-image predictors
     )
     if ret != 0:
         raise RuntimeError(f"fused_symmetric_warp_batch failed (ret={ret})")

 return image_a_prime_batch, image_b_prime_batch, self.delta_ab_pred

 Remove: The entire pc_dense_and_predictor_remap block (lines 700-735), pc_mesh_and_image_warp block (lines      
 737-760), delta_ab_dense allocation, cached_dense_maps usage, im_mesh usage.

 Key data mapping:
 - self.delta_ab_old[i, ..., 0] → pred_dy (shape nPY_prev × nPX_prev)
 - self.delta_ab_old[i, ..., 1] → pred_dx
 - self.win_ctrs_y_all[pass_idx - 1] → ctrs_y (previous pass padded centres, length nPY_prev)
 - self.win_ctrs_x_all[pass_idx - 1] → ctrs_x (length nPX_prev)

 ---
 Step 4: Ensemble Integration — cpu_ensemble.py

 5a. Load library (in _load_libraries classmethod, ~line 222)

 Follow the libmarquadt pattern — class-level cache, hard requirement:

 # Load fused warp library (required)
 fw_path = os.path.join(
     os.path.dirname(__file__), "..", "..", "lib", f"libfusedwarp{lib_extension}"
 )
 fw_path = os.path.abspath(fw_path)
 if not os.path.isfile(fw_path):
     raise FileNotFoundError(f"Required library file not found: {fw_path}")
 cls._lib_fw = ctypes.CDLL(fw_path)
 cls._lib_fw.fused_symmetric_warp_batch.restype = ctypes.c_int
 cls._lib_fw.fused_symmetric_warp_batch.argtypes = [...]  # same as instantaneous

 Add class-level default:
 _lib_fw = None

 5b. Replace _get_im_mesh + _get_image_prime_batch flow

 Ensemble architecture difference: Single predictor shared across N images. Currently:
 1. _get_im_mesh() → pad, smooth, dense remap, predictor remap, mesh construction → returns im_mesh_A,
 im_mesh_B, delta_ab_pred
 2. _get_image_prime_batch() → warp N images using im_mesh_A/B

 With fused kernel: Steps 1's dense remap + mesh construction + step 2's image warp are all replaced. The        
 remaining Python steps (padding, BCs, smoothing, predictor-to-window remap) stay.

 Implementation approach: Replace _get_im_mesh + _get_image_prime_batch with a unified method. _get_im_mesh      
 returns delta_ab_pred only (no im_mesh_A/B). A new _fused_warp_batch replaces _get_image_prime_batch.

 5c. Modified _get_im_mesh — remove dense remap + mesh construction

 _get_im_mesh now returns only delta_ab_pred (the predictor-to-window remap result). The interp_mode param
 (0=bicubic, 1=lanczos) is derived from config.

 Remove: pc_dense_remap section (lines 1335-1349), pc_mesh_construction section (lines 1398-1402),
 self.delta_ab_dense allocation, im_mesh_A/B construction.

 Keep: padding, boundary conditions, smoothing, predictor-to-window remap.

 Store window centres for use by _fused_warp_batch:

 # After smoothing, before predictor remap:
 if pass_idx > 0:
     prev_pass = pass_idx - 1
     self._fused_ctrs_y = np.ascontiguousarray(
         self.win_ctrs_y_all[prev_pass], dtype=np.float32)
     self._fused_ctrs_x = np.ascontiguousarray(
         self.win_ctrs_x_all[prev_pass], dtype=np.float32)
 else:
     self._fused_ctrs_y = np.ascontiguousarray(
         self.win_ctrs_y_all[0], dtype=np.float32)
     self._fused_ctrs_x = np.ascontiguousarray(
         self.win_ctrs_x_all[0], dtype=np.float32)

 # Predictor-to-window remap stays in Python (small grid)
 with self._profile_section(pass_idx, "pc_predictor_remap"):
     ...  # existing code unchanged

 # Map interp config → interp_mode int (0=bicubic, 1=lanczos)
 image_interp = getattr(self.config, 'ensemble_image_warp_interpolation', 'cubic')
 self._fused_interp_mode = 0 if image_interp == 'cubic' else 1

 return delta_ab_pred  # Changed return signature: no im_mesh_A/B

 5d. New _fused_warp_batch method (replaces _get_image_prime_batch)

 Single batch call to C with shared_predictor=1 — the ensemble predictor is shared across all N images. OpenMP   
 collapse(2) over (image, row) handles all parallelism internally.

 No ascontiguousarray wrapping — arrays are already float32 C-contiguous from creation. Output uses np.zeros     
 (not np.empty).

 def _fused_warp_batch(self, images_a, images_b):
     """Warp N image pairs using fused C kernel with self.delta_ab_old (shared predictor)."""
     images_a = images_a.astype(np.float32, copy=False)
     images_b = images_b.astype(np.float32, copy=False)
     N, H, W = images_a.shape
     nPY, nPX = self.delta_ab_old.shape[0], self.delta_ab_old.shape[1]

     out_a = np.zeros((N, H, W), dtype=np.float32)
     out_b = np.zeros((N, H, W), dtype=np.float32)

     ret = self._lib_fw.fused_symmetric_warp_batch(
         images_a, images_b,
         out_a, out_b,
         self.delta_ab_old[..., 0],   # pred_dy (nPY, nPX)
         self.delta_ab_old[..., 1],   # pred_dx (nPY, nPX)
         N, H, W, nPY, nPX,
         self._fused_ctrs_y, self._fused_ctrs_x,
         self._fused_interp_mode,
         1,  # shared_predictor=1 → ensemble mode
     )
     if ret != 0:
         raise RuntimeError(f"fused_symmetric_warp_batch failed (ret={ret})")

     return out_a, out_b

 5e. Update 3 call sites

 All 3 call sites (correlate_batch_for_accumulation line 671, compute_warp_sums_only line 848,
 correlate_mean_subtracted_batch line 968) change from:

 im_mesh_A, im_mesh_B, delta_ab_pred = self._get_im_mesh(pass_idx, predictor_field)
 ...
 images_a_prime, images_b_prime = self._get_image_prime_batch(
     image_a_stack, image_b_stack, im_mesh_A, im_mesh_B)

 To:

 delta_ab_pred = self._get_im_mesh(pass_idx, predictor_field)
 ...
 images_a_prime, images_b_prime = self._fused_warp_batch(
     image_a_stack, image_b_stack)

 ---
 Step 6: System Info — app.py (line 1713)

 Add "libfusedwarp" to the library check loop:

 for lib_name in ["libbulkxcorr2d", "libinterp2custom", "libmarquadt", "libfusedwarp"]:

 ---
 Step 7: Update CLAUDE.md

 Add libfusedwarp to the C Libraries table:

 | `libfusedwarp` | `fused_warp.c` | OpenMP | Fused predictor upsample + symmetric image warp |

 Add note in profiling section about new pc_fused_warp timing section.

 ---
 Implementation Order

 0. Copy this plan into docs/fused_warp_kernel_plan.md (replace old content)
 1. fused_warp.h + fused_warp.c — add fused_symmetric_warp_batch function
 2. setup.py — add build step
 3. pyproject.toml — add package-data entries
 4. Build locally — python setup.py build → verify libfusedwarp.dll produced
 5. cpu_instantaneous.py — load lib + replace warp path
 6. cpu_ensemble.py — load lib + replace warp path
 7. app.py — system_info update
 8. Run profiler to validate correctness and measure speedup
 9. CLAUDE.md — document

 ---
 Verification

 1. Build: python setup.py build produces libfusedwarp.dll (or .so) in pivtools_cli/lib/
 2. System info: GET /system_info shows libfusedwarp: {found: true}
 3. Instantaneous profiler: python profile/profile_piv.py 4mp — verify identical velocity fields, measure        
 speedup on predictor_corrector section
 4. Ensemble profiler: python profile/profile_ensemble_correlation.py 4mp --pairs 10 — verify identical
 correlation planes, measure speedup on predictor_corrector section
 5. Expected speedup: ~3x on predictor-corrector step, translating to ~15-25% overall per-pass improvement       
 depending on image size