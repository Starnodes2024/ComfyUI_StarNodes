"""
Star Nodes — Shared progress bar helpers.

Provides:
  - make_event_cb(unique_id)  → websocket callback for the DOM progress bar
  - ProgressReporter          → feeds console + ComfyUI UI + DOM progress bar
  - patch_model_for_progress  → hooks into model sampling to report per-step progress

Used by StarSampler, StarSDUpscaleRefiner, StarSDUpscaleRefinerAdvanced,
and the video tools pack.
"""

import sys
import time

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


# ---------------------------------------------------------------------------
# console bar
# ---------------------------------------------------------------------------

def _console_bar(pct, width=28):
    filled = int(round(width * min(pct, 100.0) / 100.0))
    if filled >= width:
        return "=" * width
    return "=" * filled + ">" + "-" * (width - filled - 1)


# ---------------------------------------------------------------------------
# websocket event callback
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ProgressReporter — console + ComfyUI UI + DOM events
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Feeds progress to console, ComfyUI UI ProgressBar, and DOM events."""

    def __init__(self, total_units, label="", event_cb=None):
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

        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(abs_units, self.total_units))
            except Exception:
                pass

        now = time.time()
        if now - self._last_print >= 0.25 or fraction >= 1.0:
            self._last_print = now
            pct = fraction * 100.0
            line = (f"\r[StarNodes] {self.label} "
                    f"[{_console_bar(pct)}] {pct:5.1f}%")
            if cur_s is not None and total_s:
                line += f" | {cur_s:5.1f}s/{total_s:.1f}s"
            if fps and fps not in ("0", "0.00", "N/A"):
                line += f" | {fps} fps"
            if speed and speed != "N/A":
                line += f" | {speed}"
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
        sys.stdout.write(f"\r[StarNodes] {self.label} "
                         f"[{_console_bar(100)}] 100.0% | done in "
                         f"{elapsed:.1f}s" + "    \n")
        sys.stdout.flush()
        if self.event_cb is not None:
            try:
                self.event_cb(1.0, "100%", f"done in {elapsed:.1f}s")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sampler step progress — patches model to count forward calls
# ---------------------------------------------------------------------------

def patch_model_for_progress(model, total_steps, event_cb,
                             is_flux=False, label="sampling"):
    """
    Patch a (cloned) model so that each forward pass during sampling
    reports progress to the DOM progress bar via *event_cb*.

    Returns (patched_model, reporter, cleanup_fn).

    * patched_model  — pass to the sampler instead of the original.
    * reporter       — the ProgressReporter (call reporter.finish_all(elapsed)
                       after sampling is done).
    * cleanup_fn     — call after sampling to finish the current unit.
    """
    reporter = ProgressReporter(total_units=1, label=label,
                                event_cb=event_cb)
    call_count = [0]
    # Flux / Flow models: guider calls model once per step (no uncond).
    # SD / SDXL models: CFG calls model twice per step (cond + uncond).
    calls_per_step = 1 if is_flux else 2
    total_calls = max(total_steps * calls_per_step, 1)

    def unet_wrapper(model_function, args):
        call_count[0] += 1
        frac = min(call_count[0] / total_calls, 1.0)
        current_step = min(call_count[0] // calls_per_step, total_steps)
        reporter.report(fraction=frac,
                        sub=f"step {current_step}/{total_steps}")
        return model_function(args["input"], args["timestep"],
                              **args.get("c", {}))

    patched = model.clone()
    patched.set_model_unet_function_wrapper(unet_wrapper)

    def cleanup():
        reporter.finish_unit()

    return patched, reporter, cleanup
