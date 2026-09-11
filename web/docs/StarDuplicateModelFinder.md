# Star Duplicate Model Finder

A ComfyUI custom node that scans your models directory for duplicate files based on SHA256 hash comparison.

## Features

- **Duplicate Detection**: Automatically scans all files in the selected folder and identifies duplicates by computing SHA256 hashes
- **Folder Selection**: Scan "All Models", a specific subfolder of the ComfyUI `models/` directory, or an external custom folder
- **Comprehensive Scan**: Recursively scans all subdirectories with optional filters (extensions, minimum size)
- **Output File**: Saves detailed findings to `01_ModelDuplicates.txt` in the ComfyUI output directory
- **Readable Report**: Returns a formatted multiline string and writes the same to a file, using absolute file paths
- **Sorted by Size**: Duplicate groups are sorted by file size (largest first), with size shown in MB
- **Progress & ETA**: Prints progress updates and an ETA to the console during hashing
- **JSON Cache**: Caches hashes to avoid re-hashing unchanged files; reuse is based on file size and (optionally) modified time (mtime)

## Setup

1. **No Additional Dependencies Required**: Uses only Python standard library modules
2. **Automatic Path Detection**: Automatically detects ComfyUI's models and output directories

## Inputs

### Required
- **folder**: Select the folder to scan for duplicates
  - `All Models` (default) - Scans the entire models directory
  - Or any subfolder name present in the models directory (dynamically populated)

### Optional
- **show_progress** (Boolean, default: `True`): Print progress and ETA in the ComfyUI console
- **min_size_mb** (Float, default: `10.0`): Ignore files smaller than this size (in MB)
- **extensions** (String, default: empty): Comma-separated list of file extensions to include (e.g., `.safetensors,.ckpt,.pth`)
- **include_hash** (Boolean, default: `False`): Include SHA256 hash for each duplicate group in the report
- **custom_folder_enabled** (Boolean, default: `False`): Enable scanning an external folder outside ComfyUI
- **custom_folder_path** (String, default: empty): The external folder path to scan when enabled
- **use_cache** (Boolean, default: `True`): Enable JSON caching of computed hashes
- **cache_path** (String, default: empty): Custom path for the cache JSON; default is `output/star_duplicate_cache.json`
- **validate_mtime** (Boolean, default: `True`): Cache reuse requires both size and mtime to match; if `False`, only size must match
- **prune_cache** (Boolean, default: `True`): Remove cache entries for files that no longer exist
- **rebuild_cache** (Boolean, default: `False`): Force re-hashing for the current scan; updates cache entries

## Outputs

- **report**: A formatted multiline string containing:
  - Summary of total duplicates found
  - Path to the output text file
  - Detailed list of duplicate files with their locations

## Usage Examples

### Scanning All Models
1. Set `folder` to "All Models"
2. Execute the node
3. View the report output in ComfyUI
4. Check the `01_ModelDuplicates.txt` file in your output directory for detailed information

### Scanning Specific Folder
1. Set `folder` to a specific subfolder (e.g., "checkpoints", "loras", etc.)
2. Execute the node
3. Review the duplicates found only in that folder

### Scanning a Custom Folder
1. Enable `custom_folder_enabled`
2. Set `custom_folder_path` to any directory path on your system
3. Execute the node (all other options still apply: filters, cache, etc.)

## Output Format

The node generates output in the following format:

```
Star Duplicate Finder found ## duplicates saved in file (path to txt file):
1. modelfilename (size: 2048.00 MB) found on locations:
C:/ComfyUI/models/checkpoints/model1.safetensors
C:/ComfyUI/models/loras/model1.safetensors
______________________________________________
2. another_file.safetensors (size: 1024.50 MB) [SHA256: <optional hash>] found on locations:
C:/ComfyUI/models/checkpoints/another_file.safetensors
C:/ComfyUI/models/vae/another_file.safetensors
______________________________________________
```

## Notes

- The scan may take time for large model collections due to hash computation
- Only file duplicates are detected; symbolic links and hard links are treated as separate files
- Hash computation uses SHA256 for high accuracy and collision resistance
- Files are read in 1MB chunks to handle large model files efficiently
- If no duplicates are found, the report will indicate "No duplicates found."
- The output file is always named `01_ModelDuplicates.txt` and is overwritten on each run

## Caching

- The node writes a JSON cache file to speed up subsequent scans by avoiding re-hashing unchanged files.
- Default cache location: `output/star_duplicate_cache.json` (overridable by `cache_path`).
- A cache entry stores absolute path, file size, mtime, and SHA256 hash.
- Reuse rules:
  - When `validate_mtime` is `True` (default), both size and mtime must match to reuse the hash.
  - When `False`, only size must match (useful on filesystems with unreliable mtimes).
- Maintenance:
  - `prune_cache=True` removes entries for files that no longer exist.
  - `rebuild_cache=True` forces fresh hashing for the current scan path and updates entries.

## Troubleshooting

- **"Folder does not exist"**: Ensure the selected folder exists in your models directory
- **Empty results**: Check if the selected folder contains any files
- **Slow performance**: Large model collections will take longer to scan; this is normal
- **Permission errors**: Ensure ComfyUI has read access to the models directory

## Technical Details

- Uses SHA256 hashing for duplicate detection
- Recursively walks through directory trees
- Handles large files efficiently with chunked reading
- Generates both file output and ComfyUI string output
- Integrates with ComfyUI's folder_paths system for path detection
- Thread-safe file operations with proper error handling
