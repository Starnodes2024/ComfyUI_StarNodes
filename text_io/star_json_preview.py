"""
⭐ Star JSON Preview

A ComfyUI display node that accepts a STRING containing JSON, parses it,
and renders the structure as a collapsible, syntax-highlighted tree inside
the node. The IMAGE output draws a layout mockup: when the document places
elements on an image (region text like "top left" or x/y/w/h coordinates,
as used by the Star/Ideogram scene JSON format) each element is drawn as a
colored box at its position; JSON without placement data falls back to a
nested colored-box structure diagram. The raw string is passed through
unchanged so the node can sit in the middle of a text chain. An info string
summarizes the document (root type, counts, depth, size) or the parse error.
"""

import math
import os
import re
from json import JSONDecodeError, dumps, loads

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


_SDRATIOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "json", "sdratios.json")


def _load_ratios():
    try:
        with open(_SDRATIOS_PATH, encoding="utf-8") as f:
            data = loads(f.read())["ratios"]
        return {k: (int(v["width"]), int(v["height"])) for k, v in data.items() if k != "Free Ratio"}
    except Exception:
        return {"16:9 [1344x768 landscape]": (1344, 768), "1:1 [1024x1024 square]": (1024, 1024)}


_RATIOS = _load_ratios()
DEFAULT_RATIO = "1:1 [1024x1024 square]" if "1:1 [1024x1024 square]" in _RATIOS else next(iter(_RATIOS))
TARGET_MP = 2_000_000


def _output_size(ratio_key):
    base_w, base_h = _RATIOS.get(ratio_key, _RATIOS.get(DEFAULT_RATIO))
    aspect = base_w / base_h
    w = int(math.sqrt(TARGET_MP * aspect))
    h = int(w / aspect)
    return w - w % 8, h - h % 8


def _count_nodes(value, counts, depth=0):
    counts[type(value).__name__] = counts.get(type(value).__name__, 0) + 1
    deepest = depth
    if isinstance(value, dict):
        for item in value.values():
            deepest = max(deepest, _count_nodes(item, counts, depth + 1))
    elif isinstance(value, list):
        for item in value:
            deepest = max(deepest, _count_nodes(item, counts, depth + 1))
    return deepest


def _build_info(parsed, raw):
    counts = {}
    depth = _count_nodes(parsed, counts)

    if isinstance(parsed, dict):
        root = f"object — {len(parsed)} keys"
    elif isinstance(parsed, list):
        root = f"array — {len(parsed)} items"
    else:
        root = type(parsed).__name__

    lines = [
        "Valid JSON",
        f"Root: {root}",
        f"Objects: {counts.get('dict', 0)} · Arrays: {counts.get('list', 0)} · "
        f"Strings: {counts.get('str', 0)} · Numbers: {counts.get('int', 0) + counts.get('float', 0)} · "
        f"Booleans: {counts.get('bool', 0)} · Nulls: {counts.get('NoneType', 0)}",
        f"Depth: {depth} · Size: {len(raw.encode('utf-8', 'ignore'))} bytes",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Shared drawing helpers
# ---------------------------------------------------------------------------

PAD = 12
GAP = 8
MARGIN = 14
BG = (26, 26, 46)
DIM = (122, 106, 175)
WHITE = (225, 225, 235)

# kind: (fill, border, text)
BOX_COLORS = {
    "object":    ((40, 28, 66),  (138, 90, 213),  (197, 165, 245)),
    "array":     ((18, 52, 52),  (45, 168, 158),  (138, 224, 214)),
    "string":    ((22, 48, 26),  (70, 160, 84),   (165, 214, 167)),
    "number":    ((56, 38, 14),  (214, 138, 56),  (255, 183, 77)),
    "boolean":   ((56, 24, 44),  (212, 84, 148),  (244, 143, 177)),
    "null":      ((36, 36, 42),  (104, 104, 118), (160, 160, 172)),
    "collapsed": ((34, 34, 40),  (96, 96, 110),   (150, 150, 160)),
    "error":     ((64, 18, 18),  (212, 74, 74),   (255, 138, 128)),
}


def _load_font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "Verdana.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


_FONT = _load_font(14)
_FONT_SMALL = _load_font(11)
_FONT_TITLE = _load_font(42)
_FONT_BODY = _load_font(22)
_PROBE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def _tw(text, font=_FONT):
    return _PROBE.textlength(text, font=font)


def _clip(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _literal(value):
    try:
        return _clip(dumps(value, ensure_ascii=False), 60)
    except Exception:
        return _clip(repr(value), 60)


def _kind(value):
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _badge(value):
    if isinstance(value, dict):
        return f"object · {len(value)} keys"
    if isinstance(value, list):
        return f"array · {len(value)} items"
    return None


def _hex_color(value):
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value.strip()):
        s = value.strip().lstrip("#")
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return None


def _tint(color, toward=BG, amount=0.24):
    return tuple(int(c * amount + b * (1 - amount)) for c, b in zip(color, toward))


def _mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _new_image(w, h):
    return Image.new("RGB", (int(w), int(h)), BG)


def _to_tensor(img):
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).unsqueeze(0)


# ---------------------------------------------------------------------------
#  Layout rendering — elements placed on a canvas (region text or coordinates)
# ---------------------------------------------------------------------------

REGION_KEYS = ("region", "position", "placement", "location", "area")
BOX_KEYS = ("box", "bbox", "rect", "rectangle", "bounds", "coordinates", "region", "position")
NAME_KEYS = ("name", "label", "title", "id", "type", "kind")
TEXT_KEYS = ("desc", "description", "text", "content", "caption")
PALETTE_KEYS = ("color_palette", "colors", "palette", "color")

_TOP = {"top", "upper", "topmost", "uppermost", "above"}
_BOT = {"bottom", "lower", "lowermost", "below", "base"}
_LEFT = {"left"}
_RIGHT = {"right"}
_CENTER = {"center", "centre", "middle", "mid", "centered", "centred", "central"}
_FULL = {"full", "entire", "whole", "fullscreen", "bleed", "frame", "all"}
_SMALL = {"corner", "edge", "rim", "tip"}

FALLBACK_COLORS = [
    (138, 90, 213), (45, 168, 158), (70, 160, 84), (214, 138, 56),
    (212, 84, 148), (96, 140, 220), (190, 180, 80), (150, 110, 200),
]


def _getnum(node, *keys):
    for key in keys:
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _numeric_rect(node):
    x = _getnum(node, "x", "cx", "center_x")
    y = _getnum(node, "y", "cy", "center_y")
    w = _getnum(node, "w", "width")
    h = _getnum(node, "h", "height")
    if None not in (x, y, w, h):
        return x, y, x + w, y + h
    x1 = _getnum(node, "x1", "left")
    y1 = _getnum(node, "y1", "top")
    x2 = _getnum(node, "x2", "right")
    y2 = _getnum(node, "y2", "bottom")
    if None not in (x1, y1, x2, y2) and x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    for key in BOX_KEYS:
        value = node.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4 \
                and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in value[:4]):
            x, y, a, b = (float(n) for n in value[:4])
            return x, y, x + a, y + b
    return None


def _parse_region(text):
    """Map a natural-language region like 'top left' to a normalized (x1,y1,x2,y2)."""
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    if not words:
        return None
    if words & {"background", "backdrop"}:
        return (0.0, 0.0, 1.0, 1.0)
    if words & {"foreground"}:
        return (0.0, 0.58, 1.0, 1.0)
    if words & _FULL and not (words & _TOP or words & _BOT or words & _LEFT or words & _RIGHT):
        return (0.0, 0.0, 1.0, 1.0)
    if words & {"across", "spanning", "along", "banner", "strip"}:
        if words & _TOP:
            return (0.0, 0.0, 1.0, 0.18)
        if words & _BOT:
            return (0.0, 0.82, 1.0, 1.0)
        if words & _LEFT:
            return (0.0, 0.0, 0.18, 1.0)
        if words & _RIGHT:
            return (0.82, 0.0, 1.0, 1.0)
    v = "top" if words & _TOP else "bottom" if words & _BOT else "center" if words & _CENTER else None
    h = "left" if words & _LEFT else "right" if words & _RIGHT else "center" if words & _CENTER else None
    if v is None and h is None:
        return None  # no spatial vocabulary — not a placement
    small = bool(words & _SMALL)
    if v and h:
        size = 0.26 if small else 0.34
        x1 = 0.02 if h == "left" else 1 - size - 0.02 if h == "right" else (1 - size) / 2
        y1 = 0.02 if v == "top" else 1 - size - 0.02 if v == "bottom" else (1 - size) / 2
        return (x1, y1, x1 + size, y1 + size)
    if v:
        height = 0.16 if small else 0.30
        y1 = 0.0 if v == "top" else 1 - height
        return (0.0, y1, 1.0, y1 + height)
    width = 0.16 if small else 0.30
    x1 = 0.0 if h == "left" else 1 - width
    return (x1, 0.0, x1 + width, 1.0)


def _canvas_size(parsed, fallback):
    """Find the width/height a JSON's pixel coordinates refer to, else the output frame."""
    def pair(node):
        w = _getnum(node, "width", "w")
        h = _getnum(node, "height", "h")
        if w and h and 8 <= w <= 16384 and 8 <= h <= 16384:
            return w, h
        return None
    if isinstance(parsed, dict):
        found = pair(parsed)
        if found:
            return found
        for key in ("settings", "canvas", "output", "size", "image", "dimensions", "frame"):
            sub = parsed.get(key)
            if isinstance(sub, dict):
                found = pair(sub)
                if found:
                    return found
    return fallback


def _first_string(node, keys):
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _element_color(node):
    for key in PALETTE_KEYS:
        value = node.get(key)
        if isinstance(value, list):
            for item in value:
                color = _hex_color(item)
                if color:
                    return color
        else:
            color = _hex_color(value)
            if color:
                return color
    return None


def _collect_elements(node, out):
    if isinstance(node, dict):
        rect = _numeric_rect(node)
        if rect is None:
            region = _first_string(node, REGION_KEYS)
            if region:
                rect = _parse_region(region)
        if rect is not None:
            title = _first_string(node, NAME_KEYS) or f"element {len(out) + 1}"
            body = _first_string(node, TEXT_KEYS) or ""
            if title.lower() == "text" and body:
                quote = re.search(r"'([^']{1,60})'", body)
                if quote:
                    title = f'"{quote.group(1)}"'
            out.append({"rect": rect, "node": node, "title": title, "body": body, "color": _element_color(node)})
        for value in node.values():
            _collect_elements(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_elements(value, out)


def _find_palette(parsed):
    if isinstance(parsed, dict):
        for key in ("style_description", "style", "color_palette", "palette", "colors"):
            value = parsed.get(key)
            if isinstance(value, dict):
                for sub in PALETTE_KEYS:
                    items = value.get(sub)
                    if isinstance(items, list):
                        colors = [c for c in (_hex_color(i) for i in items) if c]
                        if colors:
                            return colors
            elif isinstance(value, list):
                colors = [c for c in (_hex_color(i) for i in value) if c]
                if colors:
                    return colors
    return None


def _normalize_rect(rect, canvas_w, canvas_h):
    x1, y1, x2, y2 = rect
    if max(abs(v) for v in rect) > 1.5:  # pixel coordinates → normalize by canvas size
        x1, x2 = x1 / canvas_w, x2 / canvas_w
        y1, y2 = y1 / canvas_h, y2 / canvas_h
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 < 0.01:
        x2 = min(1.0, x1 + 0.01)
    if y2 - y1 < 0.01:
        y2 = min(1.0, y1 + 0.01)
    return x1, y1, x2, y2


def _wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, line = [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _render_layout(parsed, elements, out_w, out_h):
    canvas_w, canvas_h = _canvas_size(parsed, (float(out_w), float(out_h)))
    img = _new_image(out_w, out_h)
    draw = ImageDraw.Draw(img)

    for frac in (1 / 3, 2 / 3):  # faint rule-of-thirds grid
        draw.line([(out_w * frac, 0), (out_w * frac, out_h)], fill=_mix(BG, WHITE, 0.10))
        draw.line([(0, out_h * frac), (out_w, out_h * frac)], fill=_mix(BG, WHITE, 0.10))

    seen = {}
    for element in elements:  # cascade elements sharing one cell so all stay visible
        rect = _normalize_rect(element["rect"], canvas_w, canvas_h)
        key = (round(rect[0], 1), round(rect[1], 1))
        index = seen.get(key, 0)
        seen[key] = index + 1
        shift = index * 0.04
        element["rect"] = (min(rect[0] + shift, 0.94), min(rect[1] + shift, 0.94), rect[2], rect[3])

    for i, element in enumerate(elements):  # paint large boxes first so small ones stay readable
        element["order"] = i
    elements.sort(key=lambda e: (-(e["rect"][2] - e["rect"][0]) * (e["rect"][3] - e["rect"][1]), e["order"]))

    for i, element in enumerate(elements):
        color = element["color"] or FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        x1, y1, x2, y2 = element["rect"]
        px1, py1, px2, py2 = x1 * out_w, y1 * out_h, x2 * out_w, y2 * out_h
        draw.rounded_rectangle([px1, py1, px2, py2], radius=8, fill=_tint(color), outline=color, width=2)

        big = (py2 - py1) >= 120 and (px2 - px1) >= 200
        f_title = _FONT_TITLE if big else _FONT
        f_body = _FONT_BODY if big else _FONT_SMALL
        title_y = py1 + 10 if big else py1 + 5
        body_y = py1 + 58 if big else py1 + 24
        line_h = 27 if big else 14

        title = _clip(element["title"], 40)
        draw.text((px1 + 10, title_y), title, font=f_title, fill=_mix(color, WHITE, 0.65))
        if element["body"]:
            max_lines = max(0, int((py2 - body_y - 6) // line_h))
            for n, line in enumerate(_wrap_text(draw, _clip(element["body"], 160), f_body, px2 - px1 - 20)[:max_lines]):
                draw.text((px1 + 10, body_y + n * line_h), line, font=f_body, fill=_mix(color, WHITE, 0.35))

    palette = _find_palette(parsed)
    if palette:  # overall color palette as a swatch strip, bottom left
        for i, color in enumerate(palette[:8]):
            sx = 8 + i * 26
            draw.rounded_rectangle([sx, out_h - 30, sx + 20, out_h - 10], radius=4, fill=color, outline=WHITE)

    return img


# ---------------------------------------------------------------------------
#  Structure rendering — nested colored boxes (fallback for JSON without layout)
# ---------------------------------------------------------------------------

RENDER_BUDGET = 600   # max boxes drawn before subtrees collapse to "…" stubs
MAX_DEPTH = 8
MAX_CHILDREN = 20
HEADER_H = 26
LEAF_H = 26
CORNER = 7


def _leaf(text, kind):
    return {"kind": kind, "text": text, "w": int(_tw(text) + 2 * PAD), "h": LEAF_H}


def _measure(value, key, depth, budget):
    budget[0] -= 1
    if isinstance(value, (dict, list)):
        if depth >= MAX_DEPTH or budget[0] <= 0:
            return _leaf(f"{key}: {_badge(value)}", "collapsed")
        entries = [(str(k), v) for k, v in (value.items() if isinstance(value, dict) else enumerate(value))]
        children = []
        inner_w = inner_h = 0
        for k, v in entries[:MAX_CHILDREN]:
            if budget[0] <= 0:
                break
            child = _measure(v, _clip(k, 40), depth + 1, budget)
            children.append(child)
            inner_w = max(inner_w, child["w"])
            inner_h += child["h"] + GAP
        remaining = len(entries) - len(children)
        if remaining > 0:
            child = _leaf(f"… {remaining} more", "collapsed")
            children.append(child)
            inner_w = max(inner_w, child["w"])
            inner_h += child["h"] + GAP
        key = _clip(key, 40)
        badge = _badge(value)
        header_w = _tw(key) + (_tw(badge) + 10 if badge else 0)
        body = inner_h - GAP if children else 0
        return {"kind": _kind(value), "key": key, "badge": badge, "children": children,
                "w": int(max(inner_w, header_w) + 2 * PAD),
                "h": HEADER_H + (body + PAD if children else PAD // 2)}
    text = f"{_clip(key, 40)}: {_literal(value)}" if key else _literal(value)
    return _leaf(text, _kind(value))


def _draw_box(draw, box, x, y, w=None):
    w = box["w"] if w is None else w
    fill, border, fg = BOX_COLORS[box["kind"]]
    draw.rounded_rectangle([x, y, x + w, y + box["h"]], radius=CORNER, fill=fill, outline=border)
    if "text" in box:
        draw.text((x + PAD, y + 6), box["text"], font=_FONT, fill=fg)
        return
    draw.text((x + PAD, y + 6), box["key"], font=_FONT, fill=fg)
    if box["badge"]:
        draw.text((x + PAD + _tw(box["key"]) + 10, y + 6), box["badge"], font=_FONT, fill=DIM)
    cy = y + HEADER_H
    for child in box["children"]:
        _draw_box(draw, child, x + PAD, cy, w - 2 * PAD)
        cy += child["h"] + GAP


def _render_structure(value):
    box = _measure(value, "root", 0, [RENDER_BUDGET])
    img = _new_image(box["w"] + 2 * MARGIN, box["h"] + 2 * MARGIN)
    _draw_box(ImageDraw.Draw(img), box, MARGIN, MARGIN)
    return img


def _render_error(message):
    box = _leaf(_clip(f"Invalid JSON — {message}", 200), "error")
    img = _new_image(box["w"] + 2 * MARGIN, box["h"] + 2 * MARGIN)
    _draw_box(ImageDraw.Draw(img), box, MARGIN, MARGIN)
    return img


class StarJsonPreview:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Text And Data"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True, "tooltip": "Connect any STRING that contains JSON. The structure is rendered as a collapsible tree inside the node and as a placement-box layout on the image output."}),
                "aspect_ratio": (list(_RATIOS), {"default": DEFAULT_RATIO, "tooltip": "Frame ratio for the layout image — the same list as ⭐ Starnodes Aspect Ratio Advanced. Rendered at ~2 megapixels."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("json", "info", "image")
    OUTPUT_TOOLTIPS = (
        "The connected string, passed through unchanged.",
        "Human-readable summary: validity, root type, key/item counts, depth and size — or the parse error.",
        "Visual preview: elements drawn as colored boxes at their region/coordinates — connect to Preview Image or Save Image.",
    )
    FUNCTION = "preview"
    DESCRIPTION = (
        "Connect a STRING containing JSON and the node renders its structure as "
        "a collapsible, syntax-highlighted tree right inside the node. The image "
        "output draws a layout mockup of scene JSONs — elements become colored "
        "boxes placed by their region text (top left, background center, …) or "
        "x/y/w/h coordinates, tinted by their color palette, so you get a visual "
        "idea of the described image. JSON without placements falls back to a "
        "nested box diagram. Invalid JSON shows the exact parse error."
    )

    def preview(self, json, aspect_ratio):
        raw = json if isinstance(json, str) else str(json)
        try:
            parsed = loads(raw)
            info = _build_info(parsed, raw)
            elements = []
            _collect_elements(parsed, elements)
            image = _render_layout(parsed, elements, *_output_size(aspect_ratio)) if elements else _render_structure(parsed)
        except JSONDecodeError as error:
            info = f"Invalid JSON: {error}"
            image = _render_error(str(error))
        return {"result": (raw, info, _to_tensor(image)), "ui": {"text": [info], "star_json": [raw]}}


NODE_CLASS_MAPPINGS = {
    "StarJsonPreview": StarJsonPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarJsonPreview": "⭐ Star JSON Preview",
}
