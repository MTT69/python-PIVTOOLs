"""JSON error handlers for the Flask backend.

Every failure under ``/backend/`` must reach the browser as JSON. Without these
handlers an uncaught exception falls through to Werkzeug's HTML 500 page and the
frontend's ``res.json()`` dies with ``SyntaxError: Unexpected token '<'`` — the
real message never reaches the user.

Envelope: ``{"success": false, "error": "<message>"}`` with the real HTTP status.
Both keys are set so callers that test ``res.ok`` and callers that test
``data.success`` behave the same way.

Paths outside ``/backend/`` are left to Flask's defaults so the SPA catch-all in
``app.py`` keeps serving ``index.html``.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from loguru import logger
from werkzeug.exceptions import HTTPException, InternalServerError

API_PREFIX = "/backend/"


def _is_api_request() -> bool:
    return request.path.startswith(API_PREFIX)


def register_json_error_handlers(app: Flask) -> None:
    """Attach the two handlers to ``app``.

    ``HTTPException`` is registered first so ``abort()``, 404 and 405 keep their
    status code. Everything else becomes a 500 whose traceback goes to the loguru
    log, not to the browser.
    """

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        if not _is_api_request():
            return exc
        return jsonify({"success": False, "error": exc.description}), exc.code

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        logger.exception(
            "unhandled exception in {} {}", request.method, request.path
        )
        if not _is_api_request():
            return InternalServerError(original_exception=exc)
        return (
            jsonify({"success": False, "error": f"{type(exc).__name__}: {exc}"}),
            500,
        )
