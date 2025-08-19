from flask import Blueprint, jsonify

# Simple POD process tracking (placeholder)
pod_processing = False


POD_bp = Blueprint("POD", __name__)


@POD_bp.route("/start_pod", methods=["POST"])
def start_pod():
    """Start POD decomposition process (placeholder).

    In future this should call pod_decompose(...) from post_processing.pod_decompose
    For now, just set a flag and return 200. The real call would be something like:
        # from post_processing.pod_decompose import pod_decompose
        # thread = threading.Thread(target=pod_decompose, args=(...,), daemon=True)
        # thread.start()
    """
    # data = request.get_json(silent=True) or {}
    # base_path = data.get("base_path") or data.get("path") or data.get("full_path")
    # camera = data.get("camera", "1")
    return jsonify({"message": "Beginning POD"}), 200


@POD_bp.route("/cancel_pod", methods=["POST"])
def cancel_pod():
    """Placeholder to cancel a running POD process."""
    global pod_processing
    pod_processing = False
    return jsonify({"status": "cancelled"})


@POD_bp.route("/pod_status", methods=["GET"])
def pod_status():
    """Return a simple incremental status value for POD.

    Each call to this endpoint increases an internal counter by 5 (mod 105).
    """
    if not hasattr(pod_status, "counter"):
        pod_status.counter = 95
    pod_status.counter = (pod_status.counter + 5) % 105
    return jsonify({"status": pod_status.counter, "processing": bool(pod_processing)})
