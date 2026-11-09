import sys

sys.path.insert(0, ".")

import drift_point


def test_version():
    assert drift_point.__version__ == "0.1.0"
