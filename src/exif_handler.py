import os
from datetime import datetime
from PIL import Image

def get_exif_timestamp(img_path):
    try:
        with Image.open(img_path) as img:
            exif_data = img.getexif()
            dt_str = exif_data.get(36867)  # DateTimeOriginal
            if dt_str:
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return os.path.getmtime(img_path)

def sort_by_timeline(results):
    return sorted(results, key=lambda x: get_exif_timestamp(x[0]))
