import uuid

def get_temp_filename():
    return f"{uuid.uuid4()}"

def get_temp_filepath(ext=".xml"):
    return f"/tmp/mjcf2o3d{get_temp_filename()}{ext}"