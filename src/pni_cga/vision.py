"""Real crop acquisition, conservative verification, and optional VLM calls."""
from __future__ import annotations

import base64
from dataclasses import replace
from importlib.resources import files
from io import BytesIO
import json
import os
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps

from .execution import AcquiredView, ExecutionFailure, Verification
from .model import Action, Cost, UNKNOWN


class LocalImageBackend:
    """Only this acquisition tool opens the full image; verifiers receive crops."""
    def __init__(self, image_path: str | Path, max_edge: int = 1024):
        if max_edge < 1:
            raise ValueError("Invalid delivered crop size")
        with Image.open(image_path) as source:
            self._image = ImageOps.exif_transpose(source).convert("RGB").copy()
        self.max_edge = max_edge

    def acquire(self, action: Action) -> AcquiredView:
        if action.box is None:
            raise ExecutionFailure("Crop action requires root coordinates")
        width, height = self._image.size
        x0, y0, x1, y1 = action.box
        box = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ExecutionFailure("Empty crop after coordinate clipping")
        crop = self._image.crop(box)
        crop.thumbnail((self.max_edge, self.max_edge), Image.Resampling.BICUBIC)
        return AcquiredView(action.id, crop, box, self._image.size)


def gate_verification(payload: dict, action: Action, view: AcquiredView,
                       legibility: int = 2, binding: int = 2) -> str:
    """NO needs readable bound evidence; domain predicates additionally need scope."""
    if (not isinstance(payload, dict) or payload.get("primitive_id") != action.id
            or view.action_id != action.id):
        return UNKNOWN
    anchors = payload.get("entity_anchor_ids")
    if (not isinstance(anchors, list) or not all(isinstance(a, str) for a in anchors)
            or set(anchors) != set(action.anchor_ids)):
        return UNKNOWN
    for key, threshold in (("legibility_ordinal", legibility), ("binding_ordinal", binding)):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not threshold <= value <= 3:
            return UNKNOWN
    if action.requires_scope and payload.get("scope_complete") is not True:
        return UNKNOWN
    witnesses = payload.get("witness_regions")
    if not isinstance(witnesses, list) or not witnesses or view.root_box is None:
        return UNKNOWN
    x0, y0, x1, y1 = view.root_box
    for box in witnesses:
        if not isinstance(box, list) or len(box) != 4:
            return UNKNOWN
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) for v in box):
            return UNKNOWN
        a, b, c, d = box
        if not (x0 <= a < c <= x1 and y0 <= b < d <= y1):
            return UNKNOWN
    value = payload.get("value")
    return value.upper() if isinstance(value, str) and value.lower() in ("yes", "no") else UNKNOWN


class JSONVerifier:
    """A backend-neutral client receives only the acquired crop and primitive."""
    def __init__(self, client: Callable, legibility: int = 2, binding: int = 2):
        if legibility not in range(5) or binding not in range(5):
            raise ValueError("Gate thresholds must be ordinal values from zero to four")
        self.client, self.legibility, self.binding = client, legibility, binding

    def verify(self, view: AcquiredView, action: Action) -> Verification:
        payload, cost = self.client(view, action)
        outcome = gate_verification(payload, action, view, self.legibility, self.binding)
        return Verification(outcome, cost, "verified" if outcome != UNKNOWN else "unknown_or_rejected")


class OpenAICompatibleClient:
    """Optional actual image/model call; credentials are read from the environment.

    Implements POST /chat/completions with a JSON response. Provider-specific
    request adaptations belong in a custom client with the same return type.
    Prompt text, model and endpoint are explicit; no network calls occur at import.
    """
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 90.0, max_tokens: int = 1024):
        self.model = model or os.environ.get("PNI_CGA_MODEL", "")
        self.base_url = (base_url or os.environ.get("PNI_CGA_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("PNI_CGA_API_KEY", "")
        if not self.model or not self.base_url or not self.api_key:
            raise ValueError("Set PNI_CGA_MODEL, PNI_CGA_BASE_URL and PNI_CGA_API_KEY")
        endpoint = urlsplit(self.base_url)
        if (not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query
                or endpoint.fragment or not (endpoint.scheme == "https" or
                (endpoint.scheme == "http" and endpoint.hostname in ("localhost", "127.0.0.1", "::1")))):
            raise ValueError("Use HTTPS or a local inference endpoint")
        if not isfinite(timeout) or timeout <= 0 or isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Invalid model call limits")
        self.timeout, self.max_tokens = timeout, max_tokens
        self.prompt = files("pni_cga").joinpath("prompts/pixel_verifier.txt").read_text(encoding="utf-8")

    def __call__(self, view: AcquiredView, action: Action):
        image = BytesIO()
        view.pixels.save(image, format="PNG")
        encoded = base64.b64encode(image.getvalue()).decode("ascii")
        request_text = json.dumps({"primitive_id": action.id, "literal": action.primitive,
                                   "entity_anchor_ids": action.anchor_ids,
                                   "delivered_root_box": view.root_box,
                                   "root_image_size": view.root_size,
                                   "delivered_crop_size": view.pixels.size,
                                   "requires_scope": action.requires_scope})
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": self.prompt},
                             {"role": "user", "content": [
                                 {"type": "text", "text": request_text},
                                 {"type": "image_url", "image_url": {
                                     "url": "data:image/png;base64," + encoded}}]}]}
        request = Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + self.api_key,
                                   "Content-Type": "application/json"}, method="POST")
        start = monotonic()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            # Do not echo server responses, endpoints, or authorization headers.
            raise ExecutionFailure("Model call failed", Cost(calls=1, seconds=monotonic() - start),
                                   accounting_complete=False) from None
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if not isinstance(usage, dict) or not all(
                isinstance(usage.get(k), int) and not isinstance(usage[k], bool) and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens")):
            raise ExecutionFailure("Model response omitted token accounting",
                                   Cost(calls=1, seconds=monotonic() - start), accounting_complete=False)
        cost = Cost(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                    calls=1, seconds=monotonic() - start)
        try:
            payload = json.loads(raw["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError):
            payload = {}
        return payload, cost


class BinaryTileVerifier:
    """Offline image-demo verifier; reads only delivered black/white tile pixels."""
    def verify(self, view: AcquiredView, action: Action) -> Verification:
        pixels = np.asarray(view.pixels.convert("L"), dtype=float)
        if pixels.size == 0 or np.std(pixels) > 3:
            return Verification(UNKNOWN, reason="unreadable_tile")
        mean = float(pixels.mean())
        if 30 < mean < 225:
            return Verification(UNKNOWN, reason="unreadable_tile")
        return Verification("1" if mean >= 225 else "0")
