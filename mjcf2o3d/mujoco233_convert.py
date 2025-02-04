import mujoco


def _model_convert_mujoco233(mjcf_file, new_filename):
    print("Attempting to convert (ancient) global coordinates of MJCF file to (modern) local coordinates.")
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    mujoco.mj_saveLastXML(new_filename, model)