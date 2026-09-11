# Star Nodes — Fancy DOM Progress Bar Guide

A self-contained guide for adding the animated DOM progress bar to **any**
ComfyUI custom node. No other code from the Star Nodes pack is needed —
everything you need is in this file.

---

## Table of Contents

1. [What It Looks Like](#1-what-it-looks-like)
2. [How It Works (Architecture)](#2-how-it-works-architecture)
3. [File Structure Overview](#3-file-structure-overview)
4. [Part A — Python Side (Server)](#4-part-a--python-side-server)
5. [Part B — JavaScript Side (Frontend)](#5-part-b--javascript-side-frontend)
6. [Part C — Wiring It Into `__init__.py`](#6-part-c--wiring-it-into-__init__py)
7. [Complete Minimal Example Node](#7-complete-minimal-example-node)
8. [Customization Tips](#8-customization-tips)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What It Looks Like

The progress bar appears **inside the node itself** as a DOM widget (not the
small ComfyUI top-bar progress). It features:

- A **title row** ("Working…" / "Done") with a large percentage number.
- A **gradient-filled track** (purple → cyan) with an animated striped overlay.
- A **sub-text line** for extra info (e.g. "12.3/45.6s | 30 fps | 2.1x").
- On completion the bar turns **green** and the stripes stop animating.

```
 ┌─────────────────────────────────────────┐
 │  Working…                          42%  │
 │  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░  │
 │  12.3/45.6s | 30 fps | 2.1x             │
 └─────────────────────────────────────────┘
```

---

## 2. How It Works (Architecture)

There are **three layers** that work together:

| Layer | What | Where |
|-------|------|-------|
| **1. Python server** | Runs your long task, sends progress updates over the ComfyUI websocket via `PromptServer.send_sync()` | Your node's `.py` file |
| **2. ComfyUI websocket** | Delivers the custom event `"star_nodes.progress"` to the browser | Built into ComfyUI |
| **3. JavaScript frontend** | Listens for the event, finds the node by ID, updates the DOM widget | `web/js/your_file.js` |

**Data flow:**

```
Python node executes
  └─> ProgressReporter.report(fraction)
        └─> event_cb(overall_frac, "42%", "12.3/45.6s | 30 fps")
              └─> PromptServer.send_sync("star_nodes.progress", {node, value, text, sub})
                    └─> ComfyUI websocket
                          └─> JS: app.api.addEventListener("star_nodes.progress", ...)
                                └─> getProgressBar(node) → update DOM
```

You also get the **built-in ComfyUI top progress bar** (the one KSampler uses)
for free, because `ProgressReporter` wraps `comfy.utils.ProgressBar`.

---

## 3. File Structure Overview

For a custom node pack named `my_nodes`, you need:

```
my_nodes/
├── __init__.py              # registers nodes + WEB_DIRECTORY
├── my_node.py               # your node class (Python)
└── web/
    └── js/
        └── my_progress.js   # the frontend extension (JavaScript)
```

The `web/js/` folder is automatically served by ComfyUI when you set
`WEB_DIRECTORY = "./web"` in `__init__.py`.

---

## 4. Part A — Python Side (Server)

### A.1 — The event callback factory

This function creates a callback that pushes progress to the browser via the
ComfyUI websocket. Copy it into your node's `.py` file (or a shared utils
module):

```python
def make_event_cb(unique_id):
    """
    Build a callback that pushes progress to the node's DOM progress bar
    through the ComfyUI websocket.

    Returns None when unique_id is None or when running outside the server
    (e.g. in unit tests), so it is always safe to call.

    The callback signature is:  cb(frac, text, sub)
        frac  – float 0..1  (overall progress)
        text  – str         (big label, e.g. "42%")
        sub   – str         (small sub-text line)
    """
    if unique_id is None:
        return None
    try:
        from server import PromptServer
        srv = PromptServer.instance
    except Exception:
        return None

    def cb(frac, text, sub):
        try:
            srv.send_sync("star_nodes.progress", {
                "node": str(unique_id),
                "value": round(min(max(frac, 0.0), 1.0), 4),
                "text": text,
                "sub": sub,
            })
        except Exception:
            pass

    return cb
```

### A.2 — The `ProgressReporter` class

This class feeds progress to three places simultaneously:

1. **ComfyUI's built-in UI progress bar** (`comfy.utils.ProgressBar`)
2. **The console** (a text-based `[====>---] 42%` bar)
3. **The DOM progress bar** (via the `event_cb` callback above)

Copy this into your node's `.py` file (or a shared utils module):

```python
import sys
import time

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


def _console_bar(pct, width=28):
    """Text-based progress bar for console output."""
    filled = int(round(width * min(pct, 100.0) / 100.0))
    if filled >= width:
        return "=" * width
    return "=" * filled + ">" + "-" * (width - filled - 1)


class ProgressReporter:
    """Feeds progress to console, ComfyUI UI ProgressBar, and DOM events."""

    def __init__(self, total_units, label="", event_cb=None):
        """
        total_units – how many "units" of work (e.g. 2 for a two-pass encode,
                      or 1 for a simple single-step task).
        label       – short word shown in the console bar (e.g. "loading").
        event_cb    – the callback from make_event_cb(unique_id), or None.
        """
        self.pbar = ProgressBar(total_units) if ProgressBar else None
        self.done_units = 0.0
        self.total_units = max(total_units, 1)
        self.label = label or "working"
        self.event_cb = event_cb
        self._last_print = 0.0

    def report(self, fraction, fps=None, speed=None,
               cur_s=None, total_s=None, sub=""):
        """
        Report progress of the *current* unit.

        fraction – 0..1 progress within the current unit.
        fps/speed/cur_s/total_s – optional extra info for the sub-text line.
        sub – explicit sub-text string (overrides auto-generated sub-text).
        """
        fraction = min(max(fraction, 0.0), 1.0)
        abs_units = self.done_units + fraction

        # 1. ComfyUI built-in progress bar
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(abs_units, self.total_units))
            except Exception:
                pass

        # 2. Console bar (throttled to 4 updates/sec)
        now = time.time()
        if now - self._last_print >= 0.25 or fraction >= 1.0:
            self._last_print = now
            pct = fraction * 100.0
            line = (f"\r[MyNodes] {self.label} "
                    f"[{_console_bar(pct)}] {pct:5.1f}%")
            if cur_s is not None and total_s:
                line += f" | {cur_s:5.1f}s/{total_s:.1f}s"
            if fps and fps not in ("0", "0.00", "N/A"):
                line += f" | {fps} fps"
            if speed and speed != "N/A":
                line += f" | {speed}"
            sys.stdout.write(line + "    ")
            sys.stdout.flush()

            # 3. DOM progress bar via websocket event
            if self.event_cb is not None:
                overall = abs_units / self.total_units
                sub_parts = [p for p in (
                    sub,
                    f"{cur_s:.1f}/{total_s:.1f}s"
                        if cur_s is not None and total_s else "",
                    f"{fps} fps"
                        if fps and fps not in ("0", "0.00", "N/A") else "",
                    speed if speed and speed != "N/A" else "",
                ) if p]
                try:
                    self.event_cb(overall, f"{pct:.0f}%",
                                  " | ".join(sub_parts))
                except Exception:
                    pass

    def finish_unit(self):
        """Call after a unit (pass/file/step) is fully done."""
        self.done_units += 1.0
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(self.done_units,
                                              self.total_units))
            except Exception:
                pass

    def finish_all(self, elapsed):
        """Call when the entire task is complete."""
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(self.total_units)
            except Exception:
                pass
        sys.stdout.write(f"\r[MyNodes] {self.label} "
                         f"[{_console_bar(100)}] 100.0% | done in "
                         f"{elapsed:.1f}s" + "    \n")
        sys.stdout.flush()
        if self.event_cb is not None:
            try:
                self.event_cb(1.0, "100%", f"done in {elapsed:.1f}s")
            except Exception:
                pass
```

### A.3 — Using it in your node

In your node class:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { ... },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    def run(self, ..., unique_id=None):
        reporter = ProgressReporter(
            total_units=1,                    # 1 for a single-step task
            label="processing",
            event_cb=make_event_cb(unique_id),
        )

        t_start = time.time()
        total_steps = 100
        for i in range(total_steps):
            # ... do one step of work ...

            # Report progress (fraction 0..1 of the current unit)
            reporter.report(
                fraction=(i + 1) / total_steps,
                sub=f"step {i+1}/{total_steps}",
            )

        reporter.finish_unit()
        reporter.finish_all(time.time() - t_start)
        return (result,)
```

**Key points:**

- **`unique_id`** must be declared in `"hidden"` input types. ComfyUI
  automatically passes the node's integer ID. This is how the frontend knows
  *which* node to update.
- **`total_units`** is the number of "chunks" of work. For a single-pass
  operation, use `1`. For two-pass, use `2`. For N files, use `N` (or `2*N`
  if two-pass).
- Call **`report(fraction)`** during the work, **`finish_unit()`** after each
  unit, and **`finish_all(elapsed)`** at the very end.

---

## 5. Part B — JavaScript Side (Frontend)

Create a file at `web/js/my_progress.js` inside your node pack. This is the
**complete, self-contained** frontend code — it includes the CSS, the DOM
widget creation, and the websocket event listener.

```javascript
import { app } from "../../../../scripts/app.js";

// ─── Configuration ────────────────────────────────────────────────────────
// Add your node's class name here. The progress bar will only appear on
// nodes listed in this array.
const PROGRESS_NODES = ["MyNode"];

// Fixed height of the progress bar widget in pixels.
// ComfyUI uses widget.computedHeight for layout, NOT the live DOM height,
// so this must be set explicitly.
const PROGRESS_H = 58;

// Unique ID for the injected <style> element (prevents double-injection).
const STYLE_ID = "my-nodes-progress-style";

// ─── CSS ───────────────────────────────────────────────────────────────────
function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.my-pb { padding: 6px 8px 4px 8px; font-family: sans-serif; user-select: none; }
.my-pb-top { display: flex; justify-content: space-between; align-items: baseline;
             font-size: 11px; color: #cfcfe8; margin-bottom: 4px; }
.my-pb-pct { font-weight: 700; font-size: 13px; color: #ffffff;
             font-variant-numeric: tabular-nums; }
.my-pb-track { height: 12px; border-radius: 6px; background: #17171f;
               border: 1px solid #3a3a4c; overflow: hidden;
               box-shadow: inset 0 1px 2px rgba(0,0,0,.6); }
.my-pb-fill { height: 100%; width: 0%; border-radius: 6px; position: relative;
              background: linear-gradient(90deg, #6a5cff 0%, #00c8ff 100%);
              box-shadow: 0 0 8px rgba(0,200,255,.55);
              transition: width .15s ease-out; }
.my-pb-fill::after { content: ""; position: absolute; inset: 0;
              background: repeating-linear-gradient(45deg,
                  rgba(255,255,255,.22) 0 10px, transparent 10px 20px);
              animation: myPbStripes .8s linear infinite; }
.my-pb-sub { margin-top: 3px; font-size: 10px; color: #8a8a9e; min-height: 12px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.my-pb.done .my-pb-fill { background: linear-gradient(90deg, #2fbf71, #8ce99a);
              box-shadow: 0 0 8px rgba(80,220,130,.5); }
.my-pb.done .my-pb-fill::after { animation: none; background: none; }
@keyframes myPbStripes { from { background-position: 0 0; }
                         to   { background-position: 28px 0; } }
`;
    document.head.appendChild(st);
}

// ─── Layout helper ─────────────────────────────────────────────────────────
function relayout(node) {
    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas?.(true, true);
}

// ─── Progress bar widget ───────────────────────────────────────────────────
function getProgressBar(node) {
    // Reuse existing widget if still connected to the DOM
    if (node.myProgress && node.myProgress.wrap.isConnected) {
        return node.myProgress;
    }

    ensureStyle();

    const wrap = document.createElement("div");
    wrap.className = "my-pb";
    wrap.innerHTML =
        `<div class="my-pb-top">` +
        `<span class="my-pb-title">Working…</span>` +
        `<span class="my-pb-pct">0%</span>` +
        `</div>` +
        `<div class="my-pb-track"><div class="my-pb-fill"></div></div>` +
        `<div class="my-pb-sub"></div>`;

    const widget = node.addDOMWidget("my_progress", "myProgress", wrap, {
        serialize: false,    // don't save in workflow JSON
        hideOnZoom: false,   // keep visible when zoomed out
    });
    widget.computedHeight = PROGRESS_H;

    node.myProgress = {
        widget,
        wrap,
        title: wrap.querySelector(".my-pb-title"),
        pct:   wrap.querySelector(".my-pb-pct"),
        fill:  wrap.querySelector(".my-pb-fill"),
        sub:   wrap.querySelector(".my-pb-sub"),
    };

    relayout(node);
    return node.myProgress;
}

// ─── ComfyUI extension registration ────────────────────────────────────────
app.registerExtension({
    name: "MyNodes.ProgressBar",

    // 1. Listen for progress events from the Python server
    setup() {
        app.api.addEventListener("star_nodes.progress", (ev) => {
            const d = ev.detail || {};
            const node = app.graph.getNodeById(Number(d.node));
            if (!node) return;

            const pb = getProgressBar(node);
            const frac = Math.min(Math.max(d.value ?? 0, 0), 1);

            // Update the fill width
            pb.fill.style.width = (frac * 100).toFixed(1) + "%";

            // Update the percentage text
            pb.pct.textContent = d.text ?? Math.round(frac * 100) + "%";

            // Update the sub-text line
            pb.sub.textContent = d.sub ?? "";

            // Toggle "done" state
            if (frac >= 1) {
                pb.wrap.classList.add("done");
                pb.title.textContent = "Done";
            } else {
                pb.wrap.classList.remove("done");
                pb.title.textContent = "Working…";
            }
        });
    },

    // 2. Create the progress bar widget when a matching node is added
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PROGRESS_NODES.includes(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            getProgressBar(this);
        };
    },
});
```

### Important notes about the JS code

- **`PROGRESS_NODES`** — list the exact class name(s) from your
  `NODE_CLASS_MAPPINGS`. Only nodes listed here get the progress bar widget.
- **`addDOMWidget`** — this is a ComfyUI API that adds a DOM-based widget
  inside the node. The `serialize: false` option prevents it from being saved
  in the workflow JSON. `hideOnZoom: false` keeps it visible even when zoomed
  out.
- **`computedHeight`** — ComfyUI uses this value (not the live DOM height) to
  lay out widgets. If you change the CSS padding or font sizes, update
  `PROGRESS_H` accordingly.
- **`relayout(node)`** — after adding the widget we call
  `node.setSize(node.computeSize())` so ComfyUI recalculates the node's
  bounding box. Without this, later widgets may overlap the progress bar.
- **Event name** — `"star_nodes.progress"` must match the event name used in
  `srv.send_sync()` on the Python side. You can change it to something unique
  like `"my_nodes.progress"` — just make sure both sides match.

---

## 6. Part C — Wiring It Into `__init__.py`

Your `__init__.py` must tell ComfyUI where to find the JavaScript files:

```python
import os
import sys

# Add the node folder to sys.path so ComfyUI can import your modules
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from my_node import MyNode

NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "⭐ My Node",
}

# This is the critical line — it tells ComfyUI to serve files from ./web/
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
```

ComfyUI will automatically load every `.js` file inside `web/js/`.

---

## 7. Complete Minimal Example Node

Here is a fully working, self-contained example. It simulates a 5-second task
with progress updates, so you can see the bar in action immediately.

### `my_nodes/my_node.py`

```python
import time

# ── Progress bar helpers (copy from Section 4) ──────────────────────────
import sys

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


def _console_bar(pct, width=28):
    filled = int(round(width * min(pct, 100.0) / 100.0))
    if filled >= width:
        return "=" * width
    return "=" * filled + ">" + "-" * (width - filled - 1)


def make_event_cb(unique_id):
    if unique_id is None:
        return None
    try:
        from server import PromptServer
        srv = PromptServer.instance
    except Exception:
        return None

    def cb(frac, text, sub):
        try:
            srv.send_sync("star_nodes.progress", {
                "node": str(unique_id),
                "value": round(min(max(frac, 0.0), 1.0), 4),
                "text": text,
                "sub": sub,
            })
        except Exception:
            pass
    return cb


class ProgressReporter:
    def __init__(self, total_units, label="", event_cb=None):
        self.pbar = ProgressBar(total_units) if ProgressBar else None
        self.done_units = 0.0
        self.total_units = max(total_units, 1)
        self.label = label or "working"
        self.event_cb = event_cb
        self._last_print = 0.0

    def report(self, fraction, fps=None, speed=None,
               cur_s=None, total_s=None, sub=""):
        fraction = min(max(fraction, 0.0), 1.0)
        abs_units = self.done_units + fraction
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(abs_units, self.total_units))
            except Exception:
                pass
        now = time.time()
        if now - self._last_print >= 0.25 or fraction >= 1.0:
            self._last_print = now
            pct = fraction * 100.0
            line = (f"\r[MyNodes] {self.label} "
                    f"[{_console_bar(pct)}] {pct:5.1f}%")
            if cur_s is not None and total_s:
                line += f" | {cur_s:5.1f}s/{total_s:.1f}s"
            sys.stdout.write(line + "    ")
            sys.stdout.flush()
            if self.event_cb is not None:
                overall = abs_units / self.total_units
                sub_parts = [p for p in (
                    sub,
                    f"{cur_s:.1f}/{total_s:.1f}s"
                        if cur_s is not None and total_s else "",
                    f"{fps} fps"
                        if fps and fps not in ("0", "0.00", "N/A") else "",
                    speed if speed and speed != "N/A" else "",
                ) if p]
                try:
                    self.event_cb(overall, f"{pct:.0f}%",
                                  " | ".join(sub_parts))
                except Exception:
                    pass

    def finish_unit(self):
        self.done_units += 1.0
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(self.done_units,
                                              self.total_units))
            except Exception:
                pass

    def finish_all(self, elapsed):
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(self.total_units)
            except Exception:
                pass
        sys.stdout.write(f"\r[MyNodes] {self.label} "
                         f"[{_console_bar(100)}] 100.0% | done in "
                         f"{elapsed:.1f}s" + "    \n")
        sys.stdout.flush()
        if self.event_cb is not None:
            try:
                self.event_cb(1.0, "100%", f"done in {elapsed:.1f}s")
            except Exception:
                pass


# ── The actual node ─────────────────────────────────────────────────────
class MyNode:
    """A demo node that shows the fancy progress bar in action."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 600.0, "step": 0.1,
                    "tooltip": "How long the simulated task should take."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "MyNodes"
    OUTPUT_NODE = True

    def run(self, seconds, unique_id=None):
        reporter = ProgressReporter(
            total_units=1,
            label="demo",
            event_cb=make_event_cb(unique_id),
        )

        t_start = time.time()
        steps = max(int(seconds * 10), 1)   # 10 updates per second
        interval = seconds / steps

        for i in range(steps):
            time.sleep(interval)
            elapsed = time.time() - t_start
            reporter.report(
                fraction=(i + 1) / steps,
                cur_s=elapsed,
                total_s=seconds,
                sub=f"step {i+1}/{steps}",
            )

        reporter.finish_unit()
        reporter.finish_all(time.time() - t_start)

        return (f"Done! Simulated {seconds}s task completed.",)


NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "⭐ My Progress Demo Node",
}
```

### `my_nodes/web/js/my_progress.js`

Copy the full JavaScript code from [Section 5](#5-part-b--javascript-side-frontend).
Make sure `"MyNode"` is listed in the `PROGRESS_NODES` array.

### `my_nodes/__init__.py`

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from my_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
```

### Test it

1. Place the `my_nodes/` folder inside ComfyUI's `custom_nodes/` directory.
2. Restart ComfyUI.
3. Add the node "⭐ My Progress Demo Node" to the canvas.
4. Set `seconds` to 5 and run it.
5. Watch the progress bar animate inside the node!

---

## 8. Customization Tips

### Changing the colors

Edit the CSS in the JS file:

| CSS selector | What it controls |
|---|---|
| `.my-pb-fill` `background` | The gradient of the fill bar |
| `.my-pb-fill` `box-shadow` | The glow around the fill |
| `.my-pb.done .my-pb-fill` | The "completed" gradient (green by default) |
| `.my-pb-track` `background` | The empty track background |
| `.my-pb-pct` | The large percentage text color |
| `.my-pb-sub` | The small sub-text color |

### Changing the event name

If you have multiple node packs and want to avoid event name collisions,
change `"star_nodes.progress"` to something unique like
`"my_nodes.progress"` — but you **must** change it in **both** places:

1. **Python:** `srv.send_sync("my_nodes.progress", ...)` in `make_event_cb`
2. **JavaScript:** `app.api.addEventListener("my_nodes.progress", ...)` in `setup()`

### Multi-step tasks (more than one unit)

If your task has multiple steps (e.g. download → process → upload), set
`total_units=3` and call `finish_unit()` between each step:

```python
reporter = ProgressReporter(total_units=3, label="pipeline",
                            event_cb=make_event_cb(unique_id))

# Step 1: download
for i in range(100):
    ...
    reporter.report(fraction=(i+1)/100, sub="downloading")
reporter.finish_unit()

# Step 2: process
for i in range(100):
    ...
    reporter.report(fraction=(i+1)/100, sub="processing")
reporter.finish_unit()

# Step 3: upload
for i in range(100):
    ...
    reporter.report(fraction=(i+1)/100, sub="uploading")
reporter.finish_unit()

reporter.finish_all(time.time() - t_start)
```

The overall progress bar will smoothly go from 0% to 100% across all three
steps.

### Adjusting the widget height

If you change the CSS (e.g. larger fonts, more padding), update `PROGRESS_H`
in the JS file to match. The widget will clip or leave extra space if this
doesn't match the actual rendered height.

---

## 9. Troubleshooting

### The progress bar doesn't appear

- **Check `PROGRESS_NODES`** — the node's class name in the JS array must
  **exactly match** the key in `NODE_CLASS_MAPPINGS` (e.g. `"MyNode"`, not
  `"My Node"` or `"my_node"`).
- **Check `WEB_DIRECTORY`** — ensure `__init__.py` has
  `WEB_DIRECTORY = "./web"` and the JS file is at `web/js/your_file.js`.
- **Check browser console** (F12) — look for import errors or syntax errors
  in the JS file.
- **Restart ComfyUI** — ComfyUI caches JS extensions; a restart forces a
  reload.

### The progress bar appears but doesn't update

- **Check `unique_id`** — make sure it's declared in `"hidden"` input types
  and passed as a parameter to your function.
- **Check the event name** — the string in `send_sync()` (Python) must match
  the string in `addEventListener()` (JS).
- **Check `PromptServer` import** — if `from server import PromptServer`
  fails, the callback silently returns `None`. Check the ComfyUI server
  console for import errors.

### The progress bar overlaps other widgets

- This happens when `computedHeight` is wrong or `relayout()` isn't called.
  Make sure `PROGRESS_H` matches the actual rendered height of the widget.
- If you add **multiple** DOM widgets (e.g. progress bar + preview), always
  add the progress bar **first** so it reserves space, then add the other
  widget. Call `relayout(node)` after each addition.

### The bar is stuck at 0%

- Make sure you're calling `reporter.report(fraction)` with a value > 0.
- Make sure `fraction` is a float between 0.0 and 1.0 (not 0–100).
- Make sure `finish_all()` is called at the end — it sets the bar to 100%.

---

*This guide is self-contained and does not require any other code from the
Star Nodes pack. Everything you need is included above.*
