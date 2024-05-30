import json

import numpy as np

# Thanks to: https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return obj.item()

        return json.JSONEncoder.default(self, obj)
