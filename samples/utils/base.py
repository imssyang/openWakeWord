import numpy as np


def format_np_floats(obj, decimals=6):
    if isinstance(obj, dict):
        return {k: format_np_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_np_floats(v, decimals) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(format_np_floats(v, decimals) for v in obj)
    elif isinstance(obj, (np.float32, np.float64, float)):
        return f"{float(obj):.{decimals}f}"
    else:
        return obj
