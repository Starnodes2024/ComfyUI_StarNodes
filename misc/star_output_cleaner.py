"""
Star Output Cleaner — ComfyUI custom node.

Shows 200x200 thumbnails of the images stored in ComfyUI's output folder
(and all its subfolders) — or in a custom folder of the user's choice —
filtered by date, and lets the user delete the selected ones or download
them as a ZIP file.

Backend part: node definition + API routes
(list / thumbnail / image / browse / delete / zip).
"""

import asyncio
import calendar
import hashlib
import os
import time
import zipfile
from datetime import date, datetime, timedelta

import folder_paths
from aiohttp import web
from server import PromptServer

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is bundled with ComfyUI
    Image = None

NODE_NAME = "StarOutputCleaner"
OUTPUT_DIR = folder_paths.get_output_directory()
THUMB_SIZE = 200
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

# Thumbnail cache lives in the temp directory so it never pollutes any listing.
THUMB_CACHE_DIR = os.path.join(folder_paths.get_temp_directory(), "star_output_cleaner_thumbs")
os.makedirs(THUMB_CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _is_within(path, root):
    """True only if `path` resolves to a location inside `root`."""
    try:
        return os.path.commonpath([os.path.realpath(root), os.path.realpath(path)]) \
            == os.path.realpath(root)
    except (ValueError, OSError):
        return False


def _resolve_root(params):
    """Resolve which folder to browse from the request parameters.

    Returns (root_dir, error_message). `params` is a query dict or JSON body
    with 'folder' ("output" | "custom") and 'root' (custom absolute path).
    """
    folder = params.get("folder", "output")
    if folder == "custom":
        custom = (params.get("root", "") or "").strip()
        if not custom:
            return None, "custom folder path is empty"
        root = os.path.realpath(os.path.expanduser(custom))
        if not os.path.isdir(root):
            return None, f"folder not found: {root}"
        return root, None
    return os.path.realpath(OUTPUT_DIR), None


def _subtract_months(d, months):
    """Subtract calendar months from a date, clamping the day of month."""
    y = d.year - (months // 12)
    m = d.month - (months % 12)
    if m <= 0:
        m += 12
        y -= 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _resolve_time_range(mode, amount, unit, start_s, end_s):
    """Return (min_ts, max_ts) for the mtime filter; None = unbounded."""
    if mode == "range":
        min_ts = max_ts = None
        if start_s:
            try:
                min_ts = datetime.strptime(start_s.strip(), "%Y-%m-%d").timestamp()
            except ValueError:
                pass
        if end_s:
            try:
                # end date is inclusive -> cover the whole day
                max_ts = datetime.strptime(end_s.strip(), "%Y-%m-%d").timestamp() + 86399.999
            except ValueError:
                pass
        if min_ts is not None and max_ts is not None and min_ts > max_ts:
            min_ts, max_ts = max_ts, min_ts
        return min_ts, max_ts

    # mode "last": images from the last <amount> <unit>
    if not amount or amount <= 0:
        return None, None  # 0 = everything
    today = date.today()
    if unit == "weeks":
        start_d = today - timedelta(weeks=amount)
    elif unit == "months":
        start_d = _subtract_months(today, amount)
    else:
        start_d = today - timedelta(days=amount)
    return datetime.combine(start_d, datetime.min.time()).timestamp(), None


def _collect_images(root, min_ts, max_ts):
    """Walk `root` (and subfolders) and return images within the time range."""
    cache_root = os.path.realpath(THUMB_CACHE_DIR)
    images = []
    for dirpath, _dirs, files in os.walk(root):
        if os.path.realpath(dirpath).startswith(cache_root):
            continue
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            fpath = os.path.join(dirpath, fname)
            if not _is_within(fpath, root):
                continue
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            if min_ts is not None and st.st_mtime < min_ts:
                continue
            if max_ts is not None and st.st_mtime > max_ts:
                continue
            rel = os.path.relpath(fpath, root).replace(os.sep, "/")
            images.append({
                "path": rel,                       # path relative to the browsed root
                "name": fname,
                "subfolder": os.path.dirname(rel),
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    images.sort(key=lambda i: i["mtime"], reverse=True)  # newest first
    return images


def _int_param(params, name, default, lo=1, hi=100000):
    try:
        return max(lo, min(hi, int(params.get(name, default))))
    except (ValueError, TypeError):
        return default


routes = PromptServer.instance.routes


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #

@routes.get("/star_output_cleaner/list")
async def star_cleaner_list(request):
    """Return one page of the images in the chosen folder, date-filtered."""
    q = request.rel_url.query
    root, err = _resolve_root(q)
    if err:
        return web.json_response({"error": err, "images": []}, status=400)

    mode = q.get("mode", "last")
    unit = q.get("unit", "days")
    amount = _int_param(q, "amount", 7, lo=0)
    per_page = _int_param(q, "per_page", 100, lo=1, hi=1000)
    page = _int_param(q, "page", 1, lo=1)

    min_ts, max_ts = _resolve_time_range(mode, amount, unit,
                                         q.get("start", ""), q.get("end", ""))
    images = _collect_images(root, min_ts, max_ts)

    total = len(images)
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, pages)                      # clamp out-of-range pages
    start = (page - 1) * per_page
    return web.json_response({
        "images": images[start:start + per_page],
        "total": total,
        "page": page,
        "pages": pages,
        "per_page": per_page,
    })


@routes.get("/star_output_cleaner/thumbnail")
async def star_cleaner_thumbnail(request):
    """Return a 200x200 (letterboxed) JPEG thumbnail for the requested image."""
    q = request.rel_url.query
    root, err = _resolve_root(q)
    if err:
        return web.Response(status=400, text=err)
    rel = q.get("path", "")
    if not rel:
        return web.Response(status=400, text="missing 'path' parameter")
    fpath = os.path.join(root, rel)
    if not _is_within(fpath, root) or not os.path.isfile(fpath):
        return web.Response(status=404, text="not found")

    if Image is None:  # fallback: serve the original file
        return web.FileResponse(fpath)

    try:
        mtime = os.path.getmtime(fpath)
        key = hashlib.md5(f"{os.path.realpath(fpath)}|{mtime}".encode("utf-8")).hexdigest()
        thumb_path = os.path.join(THUMB_CACHE_DIR, key + ".jpg")
        if not os.path.isfile(thumb_path):
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (22, 22, 26))
                canvas.paste(img, ((THUMB_SIZE - img.width) // 2,
                                   (THUMB_SIZE - img.height) // 2))
                canvas.save(thumb_path, "JPEG", quality=88)
        return web.FileResponse(thumb_path, headers={"Cache-Control": "max-age=86400"})
    except Exception as exc:
        return web.Response(status=500, text=str(exc))


@routes.get("/star_output_cleaner/image")
async def star_cleaner_image(request):
    """Serve the original, full-size image (opened on double-click)."""
    q = request.rel_url.query
    root, err = _resolve_root(q)
    if err:
        return web.Response(status=400, text=err)
    rel = q.get("path", "")
    if not rel:
        return web.Response(status=400, text="missing 'path' parameter")
    fpath = os.path.join(root, rel)
    if not _is_within(fpath, root) or not os.path.isfile(fpath):
        return web.Response(status=404, text="not found")
    return web.FileResponse(fpath)  # inline, browser displays it


@routes.get("/star_output_cleaner/browse")
async def star_cleaner_browse(request):
    """List the subfolders of `path` for the folder-picker dialog.

    Only directory names are returned (never files). Hidden folders
    (starting with '.') are skipped; they can still be typed manually.
    On Windows the available drives are always included as 'drives'.
    """
    q = request.rel_url.query
    path = (q.get("path", "") or "").strip()

    drives = []
    if os.name == "nt":
        drives = [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  if os.path.isdir(f"{d}:\\")]

    # Windows with no path -> the drive list itself
    if not path and os.name == "nt":
        return web.json_response({"path": "", "parent": "", "dirs": drives,
                                  "drives": drives})

    if not path:
        path = os.path.expanduser("~")  # sensible starting point
    root = os.path.realpath(path)
    if not os.path.isdir(root):
        return web.json_response({"error": f"folder not found: {root}",
                                  "drives": drives}, status=400)

    dirs = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if entry.is_dir() and not entry.name.startswith("."):
                        dirs.append(entry.name)
                except OSError:
                    continue
    except PermissionError:
        return web.json_response({"error": f"permission denied: {root}",
                                  "drives": drives}, status=403)

    dirs.sort(key=str.lower)
    parent = os.path.dirname(root)
    if parent == root:
        parent = ""  # already at the filesystem root
    return web.json_response({"path": root, "parent": parent, "dirs": dirs,
                              "drives": drives})


@routes.post("/star_output_cleaner/delete")
async def star_cleaner_delete(request):
    """Delete the given list of image paths (relative to the browsed root)."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    root, err = _resolve_root(data)
    if err:
        return web.json_response({"error": err}, status=400)
    paths = data.get("paths", [])
    if not isinstance(paths, list):
        return web.json_response({"error": "'paths' must be a list"}, status=400)

    deleted, errors = [], []
    for rel in paths:
        rel = str(rel)
        fpath = os.path.join(root, rel)
        if not _is_within(fpath, root) or not os.path.isfile(fpath):
            errors.append({"path": rel, "error": "not found or not allowed"})
            continue
        try:
            os.remove(fpath)
            deleted.append(rel)
        except OSError as exc:
            errors.append({"path": rel, "error": str(exc)})

    return web.json_response({"deleted": deleted, "errors": errors})


@routes.post("/star_output_cleaner/zip")
async def star_cleaner_zip(request):
    """Pack the given image paths into a ZIP and send it as a download.

    Uses only the standard-library zipfile module — no extra requirements.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    root, err = _resolve_root(data)
    if err:
        return web.json_response({"error": err}, status=400)
    paths = data.get("paths", [])
    if not isinstance(paths, list) or not paths:
        return web.json_response({"error": "no paths given"}, status=400)

    files = []
    for rel in paths:
        rel = str(rel)
        fpath = os.path.join(root, rel)
        if _is_within(fpath, root) and os.path.isfile(fpath):
            files.append((rel.replace(os.sep, "/"), fpath))
    if not files:
        return web.json_response({"error": "no valid files to zip"}, status=404)

    zip_name = f"star_output_cleaner_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(folder_paths.get_temp_directory(), zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, fpath in files:
            zf.write(fpath, arcname=arcname)  # keeps the subfolder structure

    # best-effort cleanup of the temp zip 10 minutes after the download
    try:
        loop = asyncio.get_running_loop()
        loop.call_later(600, lambda p=zip_path: os.path.exists(p) and os.remove(p))
    except Exception:
        pass

    return web.FileResponse(zip_path, headers={
        "Content-Disposition": f'attachment; filename="{zip_name}"',
    })


# --------------------------------------------------------------------------- #
# node definition
# --------------------------------------------------------------------------- #

class StarOutputCleaner:
    """Utility node: all interaction happens in its custom gallery widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["output folder", "custom folder"], {
                    "default": "output folder",
                    "tooltip": "Which folder to browse: the ComfyUI output folder "
                               "or a custom folder of your choice.",
                }),
                "mode": (["last N", "date range"], {
                    "default": "last N",
                    "tooltip": "'last N': images from the last N days/weeks/months. "
                               "'date range': images between the start and end date.",
                }),
                # The inputs below are driven by the controls inside the node's
                # panel (path field, number input, unit/date pickers) and stay
                # hidden on the canvas; their values are saved with the workflow.
                "custom_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path of the folder to browse in 'custom folder' "
                               "mode. Set it via the path field / Browse button in the node.",
                }),
                "amount": ("INT", {
                    "default": 7,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "How many days/weeks/months back to show. 0 = show all images.",
                }),
                "unit": (["days", "weeks", "months"], {
                    "default": "days",
                    "tooltip": "Time unit for the 'last N' mode.",
                }),
                "start_date": ("STRING", {
                    "default": "",
                    "tooltip": "Start date YYYY-MM-DD ('date range' mode). Empty = no lower limit.",
                }),
                "end_date": ("STRING", {
                    "default": "",
                    "tooltip": "End date YYYY-MM-DD ('date range' mode). Empty = no upper limit.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    OUTPUT_NODE = True
    DESCRIPTION = ("Shows 200x200 thumbnails of the images in the output folder "
                   "(or a custom folder, incl. subfolders) — filtered by the last "
                   "N days/weeks/months or a date range — with paging, and lets you "
                   "delete the selected ones or download them as a ZIP file.")

    def noop(self, source, mode, custom_folder, amount, unit, start_date, end_date):
        return ()


NODE_CLASS_MAPPINGS = {NODE_NAME: StarOutputCleaner}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "⭐ Star Output Cleaner"}
