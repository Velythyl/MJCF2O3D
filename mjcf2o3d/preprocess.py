from pathlib import Path

import lxml.etree as etree

import mujoco
import numpy as np
import scipy.spatial.transform as transform
from scipy.spatial.transform import Rotation as R

from mjcf2o3d.cameras import generate_all_cameras
from mjcf2o3d.file_utils import get_temp_filepath


def extract_root_position_and_name(mjcf_file):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    # Find the <worldbody> element
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in the MJCF file.")

    # Find the first <body> element under <worldbody> (the root body of the robot)
    root_body = worldbody.find(".//body")
    if root_body is not None:
        root_position = root_body.get('pos', '0 0 0')
        root_name = root_body.get('name')
        print(f"Root Position: {root_position}")
    else:
        print("No root body found under <worldbody>.")
        raise AssertionError("No root body found under <worldbody>.")

    return np.array(list(map(float, root_position.split(" ")))), root_name


def remove_skylights_skyboxes_floors(mjcf_file, output_file):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    # Remove skylights, skyboxes, and floors
    for elem in root.findall(".//light[@name='skylight']"):
        parent = root.find(".//light[@name='skylight']/..")
        if parent is not None:
            parent.remove(elem)
    for elem in root.findall(".//light[@name='skybox']"):
        parent = root.find(".//light[@name='skybox']/..")
        if parent is not None:
            parent.remove(elem)
    for elem in root.findall(".//geom[@name='floor']"):
        parent = root.find(".//geom[@name='floor']/..")
        if parent is not None:
            parent.remove(elem)
    for elem in root.findall(".//geom[@type='plane']"):
        parent = root.find(".//geom[@type='plane']/..")
        if parent is not None:
            parent.remove(elem)

    # Save the modified MJCF to a new file
    tree.write(output_file)
    print(f"Modified MJCF saved to {output_file}")

def remove_visual_noise(mjcf_file, output_file, rest_of_body_colour):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    # Find all <include> elements
    #includes = root.findall(".//include")

    # Remove each <include> element
    #for include in includes:
    #    parent = include.getparent()
    #    if parent is not None:
    #        parent.remove(include)

    # Configure visual settings properly
    visual = root.find(".//visual")
    if visual is None:
        visual = etree.Element("visual")
        root.append(visual)

    # Modify materials to be non-reflective
    asset = root.find(".//asset")
    if asset is None:
        asset = etree.Element("asset")
        root.append(asset)

    # Create flat material template
    flat_material = etree.Element('material', {
        'name': 'flat_material',
        'specular': '1',
        'shininess': '1',
        'reflectance': '1',
        'rgba': rest_of_body_colour

    })
    asset.append(flat_material)

    # Apply flat material to all geoms
    for geom in root.findall(".//geom"):
        #if geom.get("material") is None:
        geom.set("material", "flat_material")

    # Save the modified MJCF
    tree.write(output_file)
    print(f"Modified MJCF saved to {output_file}")


def generate_actuator_names(mjcf_file, output_file):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    actuators = root.findall(".//actuator/*")
    name_counts = {}

    for actuator in actuators:
        if "name" not in actuator.attrib:
            joint = actuator.get("joint", "unnamed_joint")
            base_name = f"act_{joint}"

            # Ensure uniqueness
            count = name_counts.get(base_name, 0)
            new_name = base_name if count == 0 else f"{base_name}_{count}"
            name_counts[base_name] = count + 1

            actuator.set("name", new_name)

    tree.write(output_file)
    print(f"Modified MJCF with asserted actuator names saved to {output_file}")

def get_actuator_names(mjcf_file):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    # Find all actuator elements
    actuator_elements = root.findall(".//actuator/*")
    actuators = []
    for elem in actuator_elements:
        actuator_name = elem.get('name')
        actuators.append(actuator_name)
    return actuators

def remove_overlaps(mjcf_file):
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    data = mujoco.MjData(model)

    def is_overlapping(model, data):
        """Returns True if there are overlapping bodies."""
        mujoco.mj_forward(model, data)  # Update physics
        for i in range(model.nbody):
            for j in range(i + 1, model.nbody):
                if model.geom_pair[(i, j)]:  # Check if collision is enabled
                    dist = np.linalg.norm(data.xpos[i] - data.xpos[j])
                    if dist < (model.geom_size[i, 0] + model.geom_size[j, 0]):  # Check radius overlap
                        return True
        return False

    def spread_out_bodies(model, data, spread_factor=2.0):
        """Moves bodies apart until there are no overlaps."""
        for i in range(model.nbody):
            data.qpos[3 * i: 3 * i + 3] = np.random.uniform(-spread_factor, spread_factor, size=3)

        mujoco.mj_forward(model, data)
        while is_overlapping(model, data):
            for i in range(model.nbody):
                data.qpos[3 * i: 3 * i + 3] += np.random.uniform(-0.1, 0.1, size=3)  # Small random moves
            mujoco.mj_forward(model, data)

    if is_overlapping(model, data):
        spread_out_bodies(model, data)



import colorsys
def identify_actuator(mjcf_file, output_file, isolate_actuator_names, removed_actuator_colour, kept_actuator_colour, cache_name="temp"):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    # <CACHE>

    # Find all actuator elements
    actuator_elements = root.findall(".//actuator/*")
    actuators = []
    for elem in actuator_elements:
        actuator_name = elem.get('name')
        joint_name = elem.get('joint')
        if actuator_name and joint_name:
            actuators.append((actuator_name, joint_name))

    if not actuators:
        print("No actuators found in the MJCF file.")
        return

    # Helper to calculate body depth in hierarchy

    def get_body_depth(body_element):
        depth = 0
        current = body_element.getparent()
        while current is not None and current.tag == 'body':
            depth += 1
            current = current.getparent()
        return depth

    # Collect actuators with their body depth and geom information
    actuator_info = []
    for actuator_name, joint_name in actuators:
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            print(f"Warning: Joint '{joint_name}' not found for actuator '{actuator_name}'.")
            continue

        body = joint.getparent()
        if body.tag != 'body':
            print(f"Warning: Parent of joint '{joint_name}' is not a body element.")
            continue

        # Find all geoms in this body and its children
        geoms = []
        def collect_geoms(body_element):
            for geom in body_element.findall('geom'):
                geoms.append(geom)
            for child_body in body_element.findall('body'):
                collect_geoms(child_body)

        collect_geoms(body)
        if not geoms:
            print(f"Warning: No geoms found for actuator '{actuator_name}'.")
            continue

        depth = get_body_depth(body)
        actuator_info.append((depth, actuator_name, joint_name, geoms))

    # </CACHE>

    # Sort by depth (shallowest first) to process parent bodies first
    actuator_info.sort(key=lambda x: x[0])

    # Generate distinct colors for each actuator
    num_actuators = len(actuator_info)
    actuator_colors = {}
    for i, (depth, actuator_name, joint_name, geoms) in enumerate(actuator_info):
        hue = i / num_actuators  # Vary hue from 0 to 1

        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Full saturation and value

        actuator_colors[actuator_name] = (r,g,b)

        if actuator_name in isolate_actuator_names:
            rgba_str = kept_actuator_colour
        else:
            rgba_str = removed_actuator_colour
        for geom in geoms:
            geom.set('rgba', rgba_str)

    # Save the modified MJCF to a new file
    tree.write(output_file)
    print(f"Actuator-colored MJCF for {isolate_actuator_names} saved to {output_file}")
    return actuator_colors

def generate_cameras_on_sphere(center, distance, num_cameras=8):
    """
    Generate cameras evenly spaced on a sphere around a center point.

    Parameters:
    - center: numpy array of shape (3,), the center point.
    - distance: float, the distance from the center to each camera.
    - num_cameras: int, the number of cameras to generate.

    Returns:
    - cameras: list of dictionaries, each containing 'position' and 'quaternion' keys.
    """
    cameras = []

    # Generate evenly spaced points on a sphere using the Fibonacci sphere algorithm
    indices = np.arange(0, num_cameras, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_cameras)  # Polar angle
    theta = np.pi * (1 + 5 ** 0.5) * indices  # Azimuthal angle

    # Convert spherical coordinates to Cartesian coordinates
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    # Scale by distance and shift to the center
    positions = np.vstack((x, y, z)).T * distance + center

    # Calculate quaternions for each camera to point towards the center
    for position in positions:
        direction = center - position
        direction /= np.linalg.norm(direction)

        # Default forward direction is along -z
        forward = np.array([0, 0, -1])

        # Calculate the rotation axis and angle
        rotation_axis = np.cross(forward, direction)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.dot(forward, direction))

        # Create the quaternion from the axis-angle representation
        quaternion = R.from_rotvec(rotation_axis * rotation_angle).as_quat()

        # Store the camera's position and quaternion
        cameras.append({
            'position': position,
            'quaternion': quaternion
        })

    return cameras


def add_cameras_to_mjcf(mjcf_file, output_file, center, root_name, distance,isolate_actuators, num_cameras=8):
    """
    Add cameras and corresponding directional lights to the MJCF file.
    Cameras and lights are evenly spaced on a sphere around the center.
    """
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    # Find the <worldbody> element
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> element found in the MJCF file.")

    # Generate cameras
    cameras = generate_all_cameras(center, distance, num_cameras)

    if isolate_actuators:
        # Remove existing lights to avoid interference
        for light in worldbody.findall(".//light"):
            worldbody.remove(light)

    # Add cameras and lights to the <worldbody>
    for i, camera in enumerate(reversed(cameras)):  # Reverse to maintain order
        # Camera position and orientation
        position = camera['position']
#        quaternion = camera['quaternion']

        # Create camera element
        camera_element = etree.Element('camera', {
            'name': f'camera_{i}',
            'pos': " ".join(f"{x:.4f}" for x in position),
            'mode': 'targetbody',
            'target': root_name
        })
        worldbody.insert(0, camera_element)

        if isolate_actuators:
            # Create corresponding light element
            light_dir = (center - position)
            light_dir /= np.linalg.norm(light_dir)  # Normalize direction

            light_element = etree.Element('light', {
                'name': f'camera_light_{i}',
                'directional': 'true',
                'castshadow': 'false',
                'diffuse': '0.7 0.7 0.7',  # Soft white light
                'specular': '0 0 0',  # No specular highlights
                'pos': " ".join(f"{x:.4f}" for x in position),
                'dir': " ".join(f"{x:.4f}" for x in light_dir),
                'attenuation': '0 0 1'  # No distance attenuation
            })
            worldbody.insert(0, light_element)

    if isolate_actuators:
        # Add ambient light to fill in shadows
        ambient_light = etree.Element('light', {
            'name': 'ambient_light',
            'directional': 'false',
            'castshadow': 'false',
            'diffuse': '0.3 0.3 0.3',
            'specular': '0 0 0',
            'pos': "0 0 0"
        })
        worldbody.insert(0, ambient_light)

    # Save the modified MJCF to a new file
    tree.write(output_file)
    print(f"MJCF file with cameras and lights saved to {output_file}")


def _get_robot_bounding_box(model, mjcf_file):
    """
    Compute the bounding box of the robot using positions, orientations, and geom_rbound.
    """
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

    min_bound = np.array([np.inf, np.inf, np.inf])
    max_bound = np.array([-np.inf, -np.inf, -np.inf])

    def traverse_body(body_element, parent_pos, parent_quat):
        nonlocal min_bound, max_bound

        # Parse body's position and quaternion
        body_pos = np.array(list(map(float, body_element.get("pos", "0 0 0").split())))
        quat_str = body_element.get("quat", "1 0 0 0")
        quat_parts = list(map(float, quat_str.split()))
        if len(quat_parts) != 4:
            quat_parts = [1.0, 0.0, 0.0, 0.0]
        body_quat = np.array([quat_parts[1], quat_parts[2], quat_parts[3], quat_parts[0]])

        # Handle zero quaternion
        if np.linalg.norm(body_quat) < 1e-6:
            body_quat = np.array([0.0, 0.0, 0.0, 1.0])

        parent_rot = transform.Rotation.from_quat(parent_quat)
        body_pos_global = parent_rot.apply(body_pos) + parent_pos
        body_rot = parent_rot * transform.Rotation.from_quat(body_quat)

        # Process geoms
        for geom_element in body_element.findall("geom"):
            geom_pos = np.array(list(map(float, geom_element.get("pos", "0 0 0").split())))
            geom_quat_str = geom_element.get("quat", "1 0 0 0")
            geom_quat_parts = list(map(float, geom_quat_str.split()))
            if len(geom_quat_parts) != 4:
                geom_quat_parts = [1.0, 0.0, 0.0, 0.0]
            geom_quat = np.array([geom_quat_parts[1], geom_quat_parts[2], geom_quat_parts[3], geom_quat_parts[0]])

            if np.linalg.norm(geom_quat) < 1e-6:
                geom_quat = np.array([0.0, 0.0, 0.0, 1.0])

            geom_pos_global = body_rot.apply(geom_pos) + body_pos_global
            geom_rot = body_rot * transform.Rotation.from_quat(geom_quat)

            geom_name = geom_element.get("name")
            if geom_name is None:
                continue
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                continue

            rbound = model.geom_rbound[geom_id]
            min_bound = np.minimum(min_bound, geom_pos_global - rbound)
            max_bound = np.maximum(max_bound, geom_pos_global + rbound)

        # Traverse child bodies
        for child in body_element.findall("body"):
            traverse_body(child, body_pos_global, body_rot.as_quat())

    # Start traversal from root body (under worldbody)
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in the MJCF file.")
    root_body = worldbody.find("body")
    if root_body is None:
        raise ValueError("No root body found under <worldbody>.")

    traverse_body(root_body, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    return {'min': min_bound, 'max': max_bound}

def model_convert_mujoco233(mjcf_file):
    import subprocess
    path = "/".join(__file__.split("/")[:-1])

    new_filename = get_temp_filepath()
    subprocess.run('PYTHONPATH="${PYTHONPATH}:' + f'{path}" ' + f'{path}/../venv_mujoco233/bin/python -c "import mujoco233_convert; mujoco233_convert._model_convert_mujoco233(\'{mjcf_file}\', \'{new_filename}\')"', shell=True, check=True)
    return new_filename


def preprocess(mjcf_file, num_cameras=8, isolate_actuators=False):
    root_position, root_name = extract_root_position_and_name(mjcf_file)

    # Compute bounding box
    def try_load_xml(mjcf_file):
        try:
            return mjcf_file, mujoco.MjModel.from_xml_path(mjcf_file)
        except ValueError:
            pass
        except Exception as e:
            raise Exception

        mjcf_file = model_convert_mujoco233(mjcf_file)
        model = mujoco.MjModel.from_xml_path(mjcf_file)
        return mjcf_file, model

    mjcf_file, model = try_load_xml(mjcf_file)
    bounding_box = _get_robot_bounding_box(model, mjcf_file)
    size = bounding_box['max'] - bounding_box['min']
    diagonal = np.linalg.norm(size)
    camera_distance = diagonal * 3.0

    #remove_overlaps(mjcf_file)

    mjcf_file_with_actuators = get_temp_filepath()
    generate_actuator_names(mjcf_file, mjcf_file_with_actuators)
    mjcf_file = mjcf_file_with_actuators

    file_no_planes= get_temp_filepath()
    remove_skylights_skyboxes_floors(mjcf_file, file_no_planes)

    output_file = get_temp_filepath()
    add_cameras_to_mjcf(file_no_planes, output_file, root_position, root_name, camera_distance,
                        isolate_actuators=False, num_cameras=num_cameras)
    return output_file, num_cameras+6, camera_distance, root_position

def handle_actuator(workfile, actuator_names, rest_of_body_colour="t", removed_actuator_colour="t", kept_actuator_colour="w"):

    def marshall_colour(colour):
        colour = colour.lower()
        if colour in ["t", "transparent"]:
            return "0 0 0 0"
        if colour in ["w", "white"]:
            return "1 1 1 1"
        if colour in ["b", "black"]:
            return "0 0 0 1"
        return colour

    rest_of_body_colour = marshall_colour(rest_of_body_colour)
    removed_actuator_colour = marshall_colour(removed_actuator_colour)
    kept_actuator_colour = marshall_colour(kept_actuator_colour)

    file_no_objects = get_temp_filepath()
    remove_visual_noise(workfile, file_no_objects, rest_of_body_colour)

    identified_actuators = get_temp_filepath()
    actuator_colours = identify_actuator(file_no_objects, identified_actuators, actuator_names, removed_actuator_colour, kept_actuator_colour, cache_name=workfile)

    return identified_actuators, actuator_colours
