from flask import Flask, request, jsonify
from io import BytesIO
from config import Config
from image_handling.load_images import read_pair
from pre_procesing.filters import pod_filter
import numpy as np
from PIL import Image
import base64
import dask.array as da
import threading
from flask_cors import CORS
from dask import config as dask_config

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
    if camera is None or idx is None:
        return jsonify({'error': 'camera and idx required'}), 400
    # Only allow cameras listed in config.cameras
    if camera not in getattr(config, 'cameras', []):
        return jsonify({'error': f'Invalid camera name. Must be one of: {getattr(config, "cameras", [])}'}), 400
    camera_path = config.base_path / camera
    try:
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

# @app.route('/process_pod', methods=['POST'])
# def process_pod():
#     """
#     Expects JSON body:
#       - camera: camera folder name
#       - start_idx: first image index (int)
#       - count: number of pairs to process (int)
#     Returns: { status: 'processing' or 'done' }
#     """
#     global processing
#     if processing:
#         return jsonify({'status': 'processing', 'message': 'Processing already in progress'}), 409
#     data = request.get_json()
#     camera = data.get('camera')
#     start_idx = data.get('start_idx', 1)
#     count = data.get('count', 1)
#     if not camera:
#         return jsonify({'error': 'camera required'}), 400
#     camera_path = config.base_path / camera
#     indices = list(range(start_idx, start_idx + count))
#     def load_pairs():
#         pairs = [read_pair(idx, camera_path, config) for idx in indices]
#         arr = np.stack(pairs, axis=0)  # shape (N, 2, H, W)
#         darr = da.from_array(arr, chunks=(config.piv_chunk_size, 2, *config.image_shape))
#         return darr
#     def process_and_store():
#         global processing
#         darr = load_pairs()
#         processed = pod_filter(darr).compute()
#         processed_store['original'] = darr.compute()
#         processed_store['processed'] = processed
#         processing = False
#     processing = True
#     thread = threading.Thread(target=process_and_store)
#     thread.start()
#     return jsonify({'status': 'processing'})

# @app.route('/get_processed_pair', methods=['GET'])
# def get_processed_pair():
#     """
#     Query params:
#       - idx: index within processed batch (0-based)
#       - type: 'original' or 'processed'
#     Returns: PNGs as base64 in JSON
#     """
#     idx = request.args.get('idx', type=int)
#     typ = request.args.get('type', 'processed')
#     if typ not in ['original', 'processed'] or idx is None:
#         return jsonify({'error': 'idx and valid type required'}), 400
#     arr = processed_store.get(typ)
#     if arr is None or idx < 0 or idx >= arr.shape[0]:
#         return jsonify({'error': 'invalid idx or type, or processing not finished'}), 400
#     img_a = arr[idx, 0]
#     img_b = arr[idx, 1]
#     png_a = numpy_to_png_bytes(img_a)
#     png_b = numpy_to_png_bytes(img_b)
#     b64_a = base64.b64encode(png_a).decode('utf-8')
#     b64_b = base64.b64encode(png_b).decode('utf-8')
#     return jsonify({'A': b64_a, 'B': b64_b})

# Endpoint to check processing status
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'processing': processing})

if __name__ == '__main__':
    app.run(debug=True)

