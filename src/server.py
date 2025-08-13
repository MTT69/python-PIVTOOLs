from flask import Flask, request, jsonify
from io import BytesIO
from config import Config
from image_handling.load_images import read_pair
from pre_processing.filters import filter_images  # use full filter pipeline
import numpy as np
from PIL import Image
import base64
import dask.array as da
import threading
from flask_cors import CORS
from dask import config as dask_config
from pathlib import Path
from scipy.io import loadmat
from plotting.plot_maker import plot_scalar_field, make_scalar_settings
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # enable CORS for frontend dev (Next.js on a different port)

dask_config.set(scheduler='threads')

config = Config()

# In-memory storage for processed results and processing status
processed_store = {'original': None, 'processed': None}
processing = False


# Utility to convert numpy array to PNG bytes
def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


@app.route('/get_frame_pair', methods=['GET'])
def get_frame_pair():
    camera = request.args.get('camera')
    idx = request.args.get('idx', type=int)
    print(f"Received request for camera: {camera}, idx: {idx}")
    if camera is None or idx is None:
        print("Missing camera or idx in request")
        return jsonify({'error': 'camera and idx required'}), 400
    camera_path = config.source_paths[0] / camera
    try:
        print(f"Reading images from: {camera_path}")
        pair = read_pair(idx, camera_path, config)
        img_a, img_b = pair[0], pair[1]
        png_a = numpy_to_png_bytes(img_a)
        png_b = numpy_to_png_bytes(img_b)
        b64_a = base64.b64encode(png_a).decode('utf-8')
        b64_b = base64.b64encode(png_b).decode('utf-8')

        # Build response with PNGs
        resp = {'A': b64_a, 'B': b64_b}

        # Optionally include raw + meta if dtype supported (uint8/uint16)
        dtype_str = None
        bit_depth = None
        if pair.dtype == np.uint16:
            dtype_str, bit_depth = 'uint16', 16
        elif pair.dtype == np.uint8:
            dtype_str, bit_depth = 'uint8', 8

        if dtype_str is not None:
            H, W = int(pair.shape[1]), int(pair.shape[2])
            resp['meta'] = {
                'width': W,
                'height': H,
                'bitDepth': bit_depth,
                'dtype': dtype_str,
            }
            resp['A_raw'] = base64.b64encode(img_a.tobytes()).decode('utf-8')
            resp['B_raw'] = base64.b64encode(img_b.tobytes()).decode('utf-8')

        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/filter', methods=['POST'])
def filter_images_endpoint():
    """
    Expects JSON body:
      - camera: camera folder name
      - start_idx: first image index (int)
      - count: number of pairs to process (int)
      - filters: list of filter dicts (as in config.yaml), e.g.
          [{"type":"time"}, {"type":"POD"}]
    Returns: { status: 'processing' } or 409 if already processing
    """
    global processing
    if processing:
        return jsonify({'status': 'processing', 'message': 'Processing already in progress'}), 409

    data = request.get_json() or {}
    camera = data.get('camera')
    start_idx = data.get('start_idx', 1)
    count = data.get('count', 1)
    filters = data.get('filters', None)

    if camera is None or count <= 0:
        return jsonify({'error': 'camera and positive count required'}), 400

    # For testing: update the backend config filters with what the frontend sends
    if isinstance(filters, list):
        # If Config.filters is a property without a setter, you can't assign to it directly.
        # Instead, update the underlying data dict.
        config.data['filters'] = filters  # WARNING: in-memory update for this server process only
        print(f"/filter: updated config.filters = {config.filters}")
    else:
        print("/filter: no filters provided; using existing config.filters")

    camera_path = config.source_paths[0] / camera
    indices = list(range(start_idx, start_idx + count))

    def load_pairs():
        pairs = [read_pair(idx, camera_path, config) for idx in indices]
        arr = np.stack(pairs, axis=0)  # shape (N, 2, H, W)
        darr = da.from_array(arr, chunks=(config.piv_chunk_size, 2, *config.image_shape))
        return darr

    def process_and_store():
        global processing
        try:
            darr = load_pairs()
            processed = filter_images(darr, config).compute()
            processed_store['original'] = darr.compute()
            processed_store['processed'] = processed
        except Exception as e:
            print(f"Error during /filter processing: {e}")
            processed_store['original'] = None
            processed_store['processed'] = None
        finally:
            processing = False

    processing = True
    thread = threading.Thread(target=process_and_store, daemon=True)
    thread.start()
    return jsonify({'status': 'processing'})


@app.route('/get_processed_pair', methods=['GET'])
def get_processed_pair():
    """
    Query params:
      - idx: index within processed batch (0-based)
      - type: 'original' or 'processed'
    Returns: PNGs as base64 in JSON
    """
    idx = request.args.get('idx', type=int)
    typ = request.args.get('type', 'processed')
    if typ not in ['original', 'processed'] or idx is None:
        return jsonify({'error': 'idx and valid type required'}), 400
    arr = processed_store.get(typ)
    if arr is None or idx < 0 or idx >= arr.shape[0]:
        return jsonify({'error': 'invalid idx or type, or processing not finished'}), 400
    img_a = arr[idx, 0]
    img_b = arr[idx, 1]
    png_a = numpy_to_png_bytes(img_a)
    png_b = numpy_to_png_bytes(img_b)
    b64_a = base64.b64encode(png_a).decode('utf-8')
    b64_b = base64.b64encode(png_b).decode('utf-8')
    return jsonify({'A': b64_a, 'B': b64_b})


# Endpoint to check processing status
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'processing': processing})


# New endpoint: render a vector field from .mat files and return a PNG (base64)
# Query params:
#   data: absolute or relative path to data .mat file containing 'piv_result'
#   coords: absolute or relative path to coordinates .mat containing 'coordinates'
#   var: 'ux' or 'uy' (which scalar to plot)
#   run: 1-based pass index (default 1)
#   lower_limit, upper_limit: optional floats for fixed color scale
#
# Response: { image: <base64-png>, meta: { run, var, width, height } }
@app.route('/plot_vector', methods=['GET'])
def plot_vector():
    data_path_str = request.args.get('data')
    coords_path_str = request.args.get('coords')
    var = request.args.get('var', 'ux')
    run = request.args.get('run', default=1, type=int)
    lower_limit = request.args.get('lower_limit', type=float)
    upper_limit = request.args.get('upper_limit', type=float)
    cmap = request.args.get('cmap', default=None, type=str)
    print(cmap)
    if cmap is not None and (cmap.lower() == 'default' or cmap.strip() == ''):
        cmap = None

    if var not in ('ux', 'uy'):
        return jsonify({'error': "var must be 'ux' or 'uy'"}), 400
    if not data_path_str or not coords_path_str:
        return jsonify({'error': 'data and coords query params are required'}), 400

    data_path = Path(data_path_str)
    coords_path = Path(coords_path_str)
    if not data_path.exists():
        return jsonify({'error': f'data file not found: {data_path}'}), 400
    if not coords_path.exists():
        return jsonify({'error': f'coords file not found: {coords_path}'}), 400

    try:
        # Load data .mat (expects variable 'piv_result')
        data_mat = loadmat(str(data_path), struct_as_record=False, squeeze_me=True)
        if 'piv_result' not in data_mat:
            return jsonify({'error': "Variable 'piv_result' not found in data mat"}), 400
        piv_result = data_mat['piv_result']

        # Select pass element (1-based to 0-based)
        pr = None
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            if run < 1 or run > piv_result.size:
                return jsonify({'error': f'run out of range (1..{piv_result.size})'}), 400
            pr = piv_result[run - 1]
        else:
            # Single run; only valid run is 1
            if run != 1:
                return jsonify({'error': 'data contains a single run; use run=1'}), 400
            pr = piv_result

        # Extract variable and mask
        try:
            var_arr = np.asarray(getattr(pr, var))
        except Exception:
            return jsonify({'error': f"'{var}' not found in piv_result element"}), 400
        try:
            mask_arr = np.asarray(getattr(pr, 'b_mask')).astype(bool)
        except Exception:
            mask_arr = np.zeros_like(var_arr, dtype=bool)

        # Load coordinates .mat (expects variable 'coordinates')
        coords_mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        if 'coordinates' not in coords_mat:
            return jsonify({'error': "Variable 'coordinates' not found in coords mat"}), 400
        coords = coords_mat['coordinates']

        cx = cy = None
        if isinstance(coords, np.ndarray) and coords.dtype == object:
            if run < 1 or run > coords.size:
                return jsonify({'error': f'run out of range for coordinates (1..{coords.size})'}), 400
            c_el = coords[run - 1]
            cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
        else:
            if run != 1:
                return jsonify({'error': 'coordinates contains a single run; use run=1'}), 400
            c_el = coords
            cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)

        # Build settings and plot
        save_basepath = Path('plot_vector_tmp')  # not used for saving here
        settings = make_scalar_settings(
            config,
            variable=var,
            run_label=run,
            save_basepath=save_basepath,
            variable_units='m/s',
            coords_x=cx,
            coords_y=cy,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            cmap=cmap,
        )

        fig, ax, im = plot_scalar_field(var_arr, mask_arr, settings)

        # Render to PNG bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode('utf-8')

        # Optionally include dimensions
        H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
        return jsonify({'image': b64_img, 'meta': {'run': run, 'var': var, 'width': W, 'height': H}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
