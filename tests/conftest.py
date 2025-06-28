import os
import sys

# headless Linux (CI, remote machine): render offscreen with EGL. On macOS and on
# desktops with a display, MuJoCo's default backend is used.
if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
