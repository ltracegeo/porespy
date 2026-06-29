import inspect
import sys

import porespy as ps
from porespy.tools import get_edt

edt = get_edt()
ps.settings.tqdm["disable"] = True


def test_walk():
    im_2d = ps.generators.blobs(shape=[50, 50], porosity=0.5)


if __name__ == "__main__":
    current_module = sys.modules[__name__]
    for name, test_fn in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            print(f"Running {name}...")
            test_fn()
