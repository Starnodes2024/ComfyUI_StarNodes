import os
import hashlib
from collections import defaultdict
from typing import List, Dict
import time
import json
import concurrent.futures


def _star_duplicate_hash_worker(path: str):
    """Top-level worker for multiprocessing: returns (path, size, mtime, sha256)."""
    try:
        st = os.stat(path)
        mtime_local = int(st.st_mtime)
        size_local = int(st.st_size)
    except Exception:
        mtime_local = -1
        size_local = -1

    hash_sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hash_sha256.update(chunk)
        hash_local = hash_sha256.hexdigest()
    except Exception:
        hash_local = None

    return path, size_local, mtime_local, hash_local

try:
    from folder_paths import models_dir, get_output_directory
except ImportError:
    # Fallbacks if not running inside ComfyUI environment
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    models_dir = os.path.abspath(models_dir)
    def get_output_directory():
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        os.makedirs(out, exist_ok=True)
        return out

class StarDuplicateModelFinder:
    @classmethod
    def INPUT_TYPES(cls):
        # Build dropdown list from subfolders of the ComfyUI models directory
        root = models_dir if os.path.isdir(models_dir) else None
        folders = ["All Models"]
        if root:
            try:
                subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                subdirs.sort(key=str.lower)
                folders += subdirs
            except Exception:
                pass
        return {
            "required": {
                "folder": (folders, {"default": "All Models"}),
            },
            "optional": {
                "show_progress": ("BOOLEAN", {"default": True}),
                "min_size_mb": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.1}),
                "extensions": ("STRING", {"default": "", "multiline": False, "placeholder": ".safetensors,.ckpt,.pth"}),
                "include_hash": ("BOOLEAN", {"default": False}),
                "custom_folder_enabled": ("BOOLEAN", {"default": False}),
                "custom_folder_path": ("STRING", {"default": "", "multiline": False}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "cache_path": ("STRING", {"default": "", "multiline": False, "placeholder": "leave empty to use default"}),
                "validate_mtime": ("BOOLEAN", {"default": True}),
                "prune_cache": ("BOOLEAN", {"default": True}),
                "rebuild_cache": ("BOOLEAN", {"default": False}),
                "workers": ("INT", {"default": 1, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "find_duplicates"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file in chunks"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return None

    def find_duplicates(
        self,
        folder: str,
        show_progress: bool = True,
        min_size_mb: float = 0.0,
        extensions: str = "",
        include_hash: bool = False,
        custom_folder_enabled: bool = False,
        custom_folder_path: str = "",
        use_cache: bool = True,
        cache_path: str = "",
        validate_mtime: bool = True,
        prune_cache: bool = True,
        rebuild_cache: bool = False,
        workers: int = 1,
    ) -> tuple:
        models_path = models_dir if os.path.isdir(models_dir) else None
        output_path = get_output_directory()
        txt_file = os.path.join(output_path, "01_ModelDuplicates.txt")

        # Determine scan path (custom takes precedence)
        scan_path = None
        if custom_folder_enabled and custom_folder_path:
            try:
                potential = os.path.abspath(os.path.expanduser(custom_folder_path))
                if os.path.isdir(potential):
                    scan_path = potential
            except Exception:
                scan_path = None
        if scan_path is None:
            if not models_path:
                report = "Models directory not found. Please verify your ComfyUI installation."
                return (report,)
            if folder == "All Models":
                scan_path = models_path
            else:
                scan_path = os.path.join(models_path, folder)

        if not os.path.exists(scan_path):
            report = f"Folder '{scan_path}' does not exist."
            return (report,)

        # Prepare filters
        min_bytes = int(max(0.0, float(min_size_mb)) * 1024 * 1024)
        ext_set = None
        if extensions and isinstance(extensions, str):
            parts = [e.strip().lower() for e in extensions.split(',') if e.strip()]
            if parts:
                # Normalize to start with dot
                ext_set = set(e if e.startswith('.') else f'.{e}' for e in parts)

        # Collect all files (respect filters)
        files = []
        for root, dirs, filenames in os.walk(scan_path):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                try:
                    if ext_set is not None:
                        if os.path.splitext(filename)[1].lower() not in ext_set:
                            continue
                    if min_bytes > 0:
                        try:
                            if os.path.getsize(filepath) < min_bytes:
                                continue
                        except Exception:
                            continue
                    files.append(filepath)
                except Exception:
                    continue

        total_files = len(files)
        if show_progress:
            print(f"Star Duplicate Finder: {total_files} files found to check!", flush=True)

        # Prepare cache
        cache_file = None
        cache_data = {"version": 1, "files": {}}
        if use_cache:
            cache_file = cache_path.strip() or os.path.join(output_path, "star_duplicate_cache.json")
            try:
                if os.path.isfile(cache_file):
                    with open(cache_file, "r", encoding="utf-8") as cf:
                        loaded = json.load(cf)
                        if isinstance(loaded, dict) and "files" in loaded:
                            cache_data = loaded
            except Exception:
                # ignore cache load errors
                pass

            # Prune entries for files that no longer exist
            if prune_cache and "files" in cache_data and isinstance(cache_data["files"], dict):
                to_delete = []
                for p in list(cache_data["files"].keys()):
                    try:
                        if not os.path.exists(p):
                            to_delete.append(p)
                    except Exception:
                        to_delete.append(p)
                if to_delete:
                    for p in to_delete:
                        cache_data["files"].pop(p, None)
                    # Save immediately after pruning to keep cache small
                    try:
                        with open(cache_file, "w", encoding="utf-8") as cf:
                            json.dump(cache_data, cf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

        # Compute hashes
        hash_to_files = defaultdict(list)
        start_time = time.time()

        # Multiprocessing path (primarily useful for initial scans or cache rebuilds)
        use_multiprocessing = workers is not None and isinstance(workers, int) and workers > 1

        if use_multiprocessing and (not use_cache or rebuild_cache):
            max_workers = max(1, min(int(workers), 8))

            def _hash_worker(path: str):
                try:
                    st = os.stat(path)
                    mtime_local = int(st.st_mtime)
                    size_local = int(st.st_size)
                except Exception:
                    mtime_local = -1
                    size_local = -1
                hash_local = self._compute_hash(path)
                return path, size_local, mtime_local, hash_local

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(_hash_worker, p): p for p in files}
                for idx, future in enumerate(concurrent.futures.as_completed(future_to_path), start=1):
                    try:
                        filepath, size, mtime, hash_val = future.result()
                    except Exception:
                        continue

                    if hash_val:
                        hash_to_files[hash_val].append(filepath)
                        if use_cache:
                            key = os.path.abspath(filepath)
                            cache_data.setdefault("files", {})[key] = {"size": size, "mtime": mtime, "hash": hash_val}

                    if show_progress and total_files > 0:
                        elapsed = time.time() - start_time
                        avg_per_file = elapsed / idx if idx > 0 else 0
                        remaining = max(0.0, (total_files - idx) * avg_per_file)
                        percent = (idx / total_files) * 100.0
                        m, s = divmod(int(remaining), 60)
                        h, m = divmod(m, 60)
                        eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                        print(f"Progress: File {idx} of {total_files}  {percent:.1f}% done. Time Left: {eta_str}", flush=True)
        else:
            for idx, filepath in enumerate(files, start=1):
                hash_val = None
                if use_cache and not rebuild_cache:
                    try:
                        stat = os.stat(filepath)
                        mtime = int(stat.st_mtime)
                        size = int(stat.st_size)
                        key = os.path.abspath(filepath)
                        entry = cache_data.get("files", {}).get(key)
                        if entry and isinstance(entry, dict):
                            if int(entry.get("size", -1)) == size and (not validate_mtime or int(entry.get("mtime", -1)) == mtime):
                                hash_val = entry.get("hash")
                        if hash_val is None:
                            hash_val = self._compute_hash(filepath)
                            if hash_val:
                                cache_data.setdefault("files", {})[key] = {"size": size, "mtime": mtime, "hash": hash_val}
                    except Exception:
                        hash_val = self._compute_hash(filepath)
                else:
                    # Rebuild path (or no cache): always compute and update cache if enabled
                    try:
                        stat = os.stat(filepath)
                        mtime = int(stat.st_mtime)
                        size = int(stat.st_size)
                    except Exception:
                        stat = None
                        mtime = -1
                        size = -1
                    hash_val = self._compute_hash(filepath)
                    if use_cache and hash_val:
                        key = os.path.abspath(filepath)
                        cache_data.setdefault("files", {})[key] = {"size": size, "mtime": mtime, "hash": hash_val}
                if hash_val:
                    hash_to_files[hash_val].append(filepath)
                # Progress output (print every 10 files or on last)
                if show_progress and total_files > 0 and (idx == 1 or idx % 1 == 0 or idx == total_files):
                    elapsed = time.time() - start_time
                    avg_per_file = elapsed / idx if idx > 0 else 0
                    remaining = max(0.0, (total_files - idx) * avg_per_file)
                    percent = (idx / total_files) * 100.0
                    # Format ETA as mm:ss
                    m, s = divmod(int(remaining), 60)
                    h, m = divmod(m, 60)
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                    print(f"Progress: File {idx} of {total_files}  {percent:.1f}% done. Time Left: {eta_str}", flush=True)

        # Find duplicates
        duplicates = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}

        # Sort duplicates by file size (largest first)
        # All files under the same hash have identical content and size; use first path for size
        sorted_dups = []
        for h, paths in duplicates.items():
            paths_sorted = sorted(paths, key=str.lower)
            try:
                size_bytes = os.path.getsize(paths_sorted[0]) if paths_sorted else 0
            except Exception:
                size_bytes = 0
            sorted_dups.append((h, paths_sorted, size_bytes))
        sorted_dups.sort(key=lambda x: x[2], reverse=True)

        # Write to file
        with open(txt_file, "w", encoding="utf-8") as f:
            num_duplicates = len(sorted_dups)
            f.write(f"Star Duplicate Finder found {num_duplicates} duplicates saved in file {txt_file}:\n")
            if sorted_dups:
                for i, (hash_val, paths, size_bytes) in enumerate(sorted_dups, 1):
                    # Use absolute file paths for user-friendly reporting
                    abs_paths = [os.path.abspath(p) for p in paths]
                    filename = os.path.basename(paths[0])
                    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                    if include_hash:
                        f.write(f"{i}. {filename} (size: {size_mb:.2f} MB) [SHA256: {hash_val}] found on locations:\n")
                    else:
                        f.write(f"{i}. {filename} (size: {size_mb:.2f} MB) found on locations:\n")
                    for abs_path in abs_paths:
                        f.write(f"{abs_path}\n")
                    f.write("______________________________________________\n")
            else:
                f.write("No duplicates found.\n")

        # Create report string
        report_lines = []
        num_duplicates = len(sorted_dups)
        report_lines.append(f"Star Duplicate Finder found {num_duplicates} duplicates saved in file {txt_file}:")
        if sorted_dups:
            for i, (hash_val, paths, size_bytes) in enumerate(sorted_dups, 1):
                abs_paths = [os.path.abspath(p) for p in paths]
                filename = os.path.basename(paths[0])
                size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                if include_hash:
                    report_lines.append(f"{i}. {filename} (size: {size_mb:.2f} MB) [SHA256: {hash_val}] found on locations:")
                else:
                    report_lines.append(f"{i}. {filename} (size: {size_mb:.2f} MB) found on locations:")
                for abs_path in abs_paths:
                    report_lines.append(f"{abs_path}")
                report_lines.append("______________________________________________")
        else:
            report_lines.append("No duplicates found.")

        report = "\n".join(report_lines)

        # Save cache if enabled
        if use_cache and cache_file:
            try:
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(cache_data, cf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return (report,)


NODE_CLASS_MAPPINGS = {
    "StarDuplicateModelFinder": StarDuplicateModelFinder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarDuplicateModelFinder": "⭐ Star Duplicate Model Finder",
}
