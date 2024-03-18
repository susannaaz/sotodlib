import numpy as np
import so3g
from spt3g import core as g3core

def fetch_hk(path, fields=None):
    hk_data = {}
    reader = so3g.G3IndexedReader(path)

    while True:
        frames = reader.Process(None)
        if not frames:
            break

        for frame in frames:
            if 'address' in frame:
                for v in frame['blocks']:
                    for k in v.keys():
                        field = '.'.join([frame['address'], k])

                        if fields is None or field in fields:
                            key = field.split('.')[-1]
                            if k == key:
                                data = [[t.time / g3core.G3Units.s for t in v.times], v[k]]
                                hk_data.setdefault(field, ([], []))
                                hk_data[field] = (
                                    np.concatenate([hk_data[field][0], data[0]]),
                                    np.concatenate([hk_data[field][1], data[1]])
                                )
    return hk_data
