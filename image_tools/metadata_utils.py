"""
StarNodes IO - metadata helpers.

Embed and read the 5 custom "StarMetaData" fields in PNG / JPG / WEBP files.
This module has NO ComfyUI imports so it can be tested standalone.
"""

import json

METADATA_BLOB_KEY = "StarMetaData"  # single JSON blob key stored in every format

STAR_METADATA_KEYS = [f"StarMetaData {i}" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# Embedding metadata into image files
# ---------------------------------------------------------------------------

def _stringify_metadata(metadata):
    """Ensure every value is a string (PNG text chunks require str)."""
    out = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            out[str(key)] = value
        elif isinstance(value, (dict, list)):
            out[str(key)] = json.dumps(value, ensure_ascii=False)
        else:
            out[str(key)] = str(value)
    return out


def build_png_info(metadata):
    """Build a PngInfo object with each entry plus the combined JSON blob."""
    from PIL.PngImagePlugin import PngInfo

    data = _stringify_metadata(metadata)
    png_info = PngInfo()
    blob = json.dumps(data, ensure_ascii=False)
    png_info.add_text(METADATA_BLOB_KEY, blob)
    for key, value in data.items():
        if key == METADATA_BLOB_KEY:
            continue
        try:
            png_info.add_text(key, value)
        except Exception:
            pass
        # ComfyUI 'comf' chunks (after IDAT) for maximum compatibility.
        try:
            png_info.add(
                b"comf",
                key.encode("latin-1", "strict") + b"\0" + value.encode("latin-1", "strict"),
                after_idat=True,
            )
        except Exception:
            pass
    return png_info


def build_exif_bytes(metadata):
    """Build EXIF bytes carrying the metadata JSON blob (JPG / WEBP)."""
    from PIL import Image

    data = _stringify_metadata(metadata)
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    exif = Image.Exif()
    # 0x9286 = UserComment. Prefix with the standard 8-byte ASCII charset marker.
    exif[0x9286] = b"ASCII\x00\x00\x00" + payload
    # 0x010E = ImageDescription, a human friendly fallback.
    try:
        summary = ", ".join(f"{k}={v[:40]}" for k, v in list(data.items())[:4])
        exif[0x010E] = f"{METADATA_BLOB_KEY}: {summary}"[:512]
    except Exception:
        pass
    return exif.tobytes()


# ---------------------------------------------------------------------------
# Reading metadata back
# ---------------------------------------------------------------------------

def _decode_bytes(value):
    if isinstance(value, bytes):
        for encoding in ("utf-8", "latin-1"):
            try:
                return value.decode(encoding)
            except Exception:
                continue
        return None
    return value


def _parse_user_comment(raw):
    """Parse an EXIF UserComment value into a metadata dict (or None)."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        # Strip the 8-byte charset prefix when present.
        if len(raw) > 8 and raw[:5] in (b"ASCII", b"UNICO", b"JIS\x00\x00"):
            raw = raw[8:]
        raw = _decode_bytes(raw)
        if raw is None:
            return None
    if not isinstance(raw, str):
        return None
    text = raw.strip().lstrip("\ufeff").strip("\x00").strip()
    if text.startswith(f"{METADATA_BLOB_KEY}:"):
        text = text.split(":", 1)[1].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Legacy "key:value" style (StarMetaInjector WebP EXIF).
    if ":" in text:
        key, value = text.split(":", 1)
        if key and value:
            return {key.strip(): value.strip()}
    return None


def read_star_metadata(pil_image):
    """Read StarNodes metadata from a PIL image (PNG / JPG / WEBP / ...).

    Returns an insertion-ordered dict. Individual text entries win, the
    combined JSON blob fills in the rest.
    """
    md = {}

    # 1) PNG text chunks / generic info dict.
    sources = {}
    text_chunks = getattr(pil_image, "text", None)
    if isinstance(text_chunks, dict):
        for key, value in text_chunks.items():
            value = _decode_bytes(value)
            if isinstance(value, str):
                sources[str(key)] = value
    info = getattr(pil_image, "info", None)
    if isinstance(info, dict):
        for key, value in info.items():
            value = _decode_bytes(value)
            if isinstance(value, str):
                sources.setdefault(str(key), value)

    blob = sources.get(METADATA_BLOB_KEY)
    if blob:
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                md.update(data)
        except Exception:
            pass

    skip_keys = {METADATA_BLOB_KEY, "parameters", "Comment", "exif", "xmp", "XML:com.adobe.xmp"}
    for key, value in sources.items():
        if key in skip_keys or key in md:
            continue
        md[key] = value

    # 2) EXIF UserComment (JPG / WEBP written by StarNodes).
    try:
        exif = pil_image.getexif()
        if exif is not None:
            parsed = _parse_user_comment(exif.get(0x9286))
            if parsed:
                for key, value in parsed.items():
                    md.setdefault(str(key), value)
            # Legacy string EXIF entries "key:value".
            for exif_tag, exif_value in exif.items():
                if exif_tag in (0x9286, 0x010E):  # UserComment / our ImageDescription
                    continue
                if isinstance(exif_value, str) and ":" in exif_value:
                    if exif_value.startswith(f"{METADATA_BLOB_KEY}:"):
                        continue
                    key, value = exif_value.split(":", 1)
                    if key and value and key not in md:
                        md.setdefault(key.strip(), value.strip())
    except Exception:
        pass

    return md
