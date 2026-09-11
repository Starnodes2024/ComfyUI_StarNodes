# StarNodes install hook
#
# This pack intentionally performs NO automatic pip installs and runs NO
# subprocesses at install time, in line with the ComfyUI Registry security
# policy. All regular dependencies are declared in requirements.txt /
# pyproject.toml and are installed by ComfyUI-Manager.
#
# The only optional, extra dependency is "nvidia-vfx" (used solely by the
# optional "Star Advanced RTX VSR" node). It is distributed via NVIDIA's
# own PyPI index and cannot be installed automatically here. If you want to
# use that node, install it manually:
#
#     pip install -U --no-build-isolation nvidia-vfx --extra-index-url https://pypi.nvidia.com
#
# Every node that needs an optional dependency fails gracefully with a clear
# console message when the dependency is missing - the rest of the pack is
# unaffected.
