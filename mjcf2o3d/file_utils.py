import json
import uuid
import gzip

import numpy as np


def get_temp_filename():
    return f"mjcf2o3d{uuid.uuid4()}"

def get_temp_filepath(ext=".xml"):
    return f"/tmp/{get_temp_filename()}{ext}"

def save_json(json_obj, outpath):
    with gzip.open(outpath, "wt", encoding="UTF-8") as f:
        json.dump(json_obj, f)

def load_json(inpath, array_func=np.asarray):
    with gzip.open(inpath, 'rt', encoding='UTF-8') as zipfile:
        x = json.load(zipfile)

    ret = {}
    for partname, subdict in x.items():
        subret = {}
        for k, v in subdict.items():
            if isinstance(v, list) or isinstance(v, tuple):
                v = array_func(v)
            subret[k] = v
        ret[partname] = subret
    return ret
