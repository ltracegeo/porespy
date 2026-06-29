import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["infer_tag"] = True  # Ensure platform-specific wheel tag

        result = subprocess.run(["meson", "setup", "--wipe", "builddir"], check=True)

        if result.returncode != 0:
            sys.exit(f"[hook] Build script failed with exit code {result.returncode}")

        result = subprocess.run(["meson", "compile", "-C", "builddir"], check=True)

        if result.returncode != 0:
            sys.exit(f"[hook] Build script failed with exit code {result.returncode}")
