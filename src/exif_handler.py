import os
from datetime import datetime
from PIL import Image
from raw_support import get_exif_timestamp as _raw_get_exif_ts

def get_exif_timestamp(img_path):
    ts = _raw_get_exif_ts(img_path)
    if ts is not None:
        return ts
    return os.path.getmtime(img_path)

def sort_by_timeline(results):
    return sorted(results, key=lambda x: get_exif_timestamp(x[0]))
