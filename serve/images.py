"""Images on the wire: base64 data URLs in, validated PIL images out.

A request carries `images`, a list of 1-4 data URLs
(`data:image/png;base64,...`; PNG, JPEG or WebP). The state refers to them with
the literal marker `<image>`, one per image, in order; a state without markers
gets its images put before it. The model sees each image at its own resolution,
scaled down so that all of a request's images together take at most the
checkpoint's image-token budget (1,024 tokens for v4.0-VL: one image up to ~1 MP,
four images up to 256 tokens each).

Every rejection here is a `RequestError` (HTTP 422) that says what to change.
Only data URLs are accepted: fetching an http(s) URL would make the server a
client of whatever address a request names.

The limits (20 MiB, 20 million pixels, single-frame PNG/JPEG/WebP) are the ones
imajev's own loader applies to uploads and data URLs.
"""
from __future__ import annotations

import base64
import binascii
import io
import re
import warnings
from typing import Any

from serve.wire import RequestError

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 20 * 2**20            # decoded bytes of one image
MAX_IMAGE_PIXELS = 20_000_000           # decoded width x height of one image
MAX_ASPECT_RATIO = 200                  # the image processor refuses anything longer
FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/jpg": "JPEG", "image/webp": "WEBP"}
MARKER = "<image>"

_DATA_URL = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)(?P<params>(;[\w.+-]+=[\w.+-]+)*);base64,",
                       re.IGNORECASE)


def image_limits(meta_vision: dict | None) -> dict[str, Any]:
    """What `GET /v1/limits` reports under `images`."""
    if not meta_vision:
        return {"supported": False}
    budget = int(meta_vision["image_token_budget"])
    floor = int(meta_vision.get("min_tokens_per_image", 64))
    return {"supported": True,
            "max_images": MAX_IMAGES,
            "image_token_budget": budget,
            "max_tokens_per_image": {str(n): max(floor, budget // n) for n in range(1, MAX_IMAGES + 1)},
            "formats": sorted({v.lower() for v in FORMATS.values()}),
            "encoding": "data URL (data:image/<png|jpeg|webp>;base64,...)",
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "state_marker": MARKER,
            "prefix_cache": False,
            "document_cache": False}


def decode_data_url(url: Any, index: int) -> bytes:
    """The bytes of one data URL, or a 422 that names the image."""
    where = f"images[{index}]"
    if not isinstance(url, str):
        raise RequestError(f"{where}: must be a data URL string")
    if url[:8].lower().startswith(("http://", "https://")):
        raise RequestError(f"{where}: only data URLs are accepted "
                           f"(data:image/png;base64,...); http(s) URLs are not fetched")
    m = _DATA_URL.match(url[:256])
    if not m:
        raise RequestError(f"{where}: not a base64 data URL; expected "
                           f"data:image/<png|jpeg|webp>;base64,<data>")
    mime = m.group("mime").lower()
    if mime not in FORMATS:
        raise RequestError(f"{where}: unsupported type {mime}; use PNG, JPEG or WebP")
    payload = url[m.end():]
    # base64 inflates by 4/3; refuse an oversized payload before decoding it.
    if len(payload) > (MAX_IMAGE_BYTES * 4) // 3 + 8:
        raise RequestError(f"{where}: image is over {MAX_IMAGE_BYTES // 2**20} MiB")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(f"{where}: invalid base64 data") from None
    if not data:
        raise RequestError(f"{where}: empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise RequestError(f"{where}: image is over {MAX_IMAGE_BYTES // 2**20} MiB")
    return data


def load_image_bytes(data: bytes, index: int):
    """Decode and validate one image; returns an RGB PIL image."""
    from PIL import Image, ImageOps
    where = f"images[{index}]"
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(data)) as src:
                if src.format not in {"PNG", "JPEG", "WEBP"}:
                    raise RequestError(f"{where}: unsupported format {src.format}; "
                                       f"use PNG, JPEG or WebP")
                if getattr(src, "n_frames", 1) != 1:
                    raise RequestError(f"{where}: animated or multi-frame images are not supported")
                w, h = src.size
                if w * h > MAX_IMAGE_PIXELS:
                    raise RequestError(f"{where}: {w}x{h} is over {MAX_IMAGE_PIXELS:,} pixels")
                if max(w, h) > MAX_ASPECT_RATIO * min(w, h):
                    raise RequestError(f"{where}: aspect ratio {w}x{h} is over "
                                       f"{MAX_ASPECT_RATIO}:1")
                src.load()
                # A camera's EXIF rotation is applied, so the model sees the photo the
                # way a viewer does. Dataset images carry none, so this is a no-op there.
                return ImageOps.exif_transpose(src).convert("RGB")
        except RequestError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise RequestError(f"{where}: image is too large to decode safely") from None
        except Exception as e:                       # PIL's UnidentifiedImageError, truncation...
            raise RequestError(f"{where}: could not decode the image ({type(e).__name__})") from None


def check_markers(state: str, n_images: int) -> None:
    """The state's `<image>` markers must match the images, or be absent."""
    from rsijev.vision import VISION_SPECIAL_TOKENS
    k = state.count(MARKER)
    if k not in (0, n_images):
        raise RequestError(f"state has {k} {MARKER} markers for {n_images} images: use one "
                           f"marker per image, in order, or none (images then go first)")
    bad = [t for t in VISION_SPECIAL_TOKENS if t in state]
    if bad:
        raise RequestError(f"state contains the reserved token {bad[0]}; refer to images "
                           f"with {MARKER}")


def parse_images(urls: list, state: str) -> list:
    """Wire `images` + rendered state -> validated PIL images (possibly none)."""
    if not urls:
        return []
    if len(urls) > MAX_IMAGES:
        raise RequestError(f"images: at most {MAX_IMAGES} per request, got {len(urls)}")
    check_markers(state, len(urls))
    return [load_image_bytes(decode_data_url(u, i), i) for i, u in enumerate(urls)]


def to_data_url(image) -> str:
    """A data URL for a PIL image, a path, raw bytes or an existing data URL.

    For `Decider`: everything goes through the server's own decoding. A PIL image
    is written as PNG, which is lossless, so the pixels the model sees are the
    pixels passed in."""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    elif hasattr(image, "save") and hasattr(image, "mode"):     # a PIL image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
    else:
        with open(image, "rb") as f:
            data = f.read()
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        fmt = (im.format or "PNG").lower()
    mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(fmt, f"image/{fmt}")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def state_too_long(e: ValueError, max_length: int) -> RequestError | None:
    """The 422 for an image request whose state would be cut into an image, or None
    when `e` is some other ValueError (which stays a 500)."""
    if "cut into an image" not in str(e) and "markers" not in str(e):
        return None
    return RequestError(
        "the state is too long to keep its images whole: the input is "
        f"capped at {max_length} tokens and a longer state is cut from "
        "the left. Shorten the state or put the <image> markers after the "
        f"text ({e})")


def text_only_error(name: str, reason: str | None = None) -> RequestError:
    return RequestError(reason or f"{name} is a text-only model; it does not take images")
