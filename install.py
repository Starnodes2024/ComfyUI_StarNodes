import sys
import subprocess
import importlib.util

if importlib.util.find_spec("nvvfx") is None:
    print("StarNodes: Installing nvidia-vfx dependency...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-U",
        "--no-build-isolation", "nvidia-vfx",
        "--extra-index-url", "https://pypi.nvidia.com"
    ])