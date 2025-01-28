import uuid

def get_temp_filename():
    return f"{uuid.uuid4()}"

def get_temp_filepath(ext=".xml"):
    return f"/tmp/{get_temp_filename()}{ext}"