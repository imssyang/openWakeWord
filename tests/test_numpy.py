import numpy as np
from source.utils import format_np_floats


def test_format_np_floats():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6]], dtype=np.float32)
    b = format_np_floats(a)
    print(b)
