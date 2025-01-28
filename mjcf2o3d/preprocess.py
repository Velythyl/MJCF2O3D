import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import scipy.spatial.transform as transform
from scipy.spatial.transform import Rotation as R

from mjcf2o3d.cameras import generate_all_cameras
from mjcf2o3d.file_utils import get_temp_filepath


def extract_root_position_and_bounding_box(mjcf_file):
    # Parse the MJCF file
    tree = ET.parse(mjcf_file)
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
    tree = ET.parse(mjcf_file)
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


def add_cameras_to_mjcf(mjcf_file, output_file, center, root_name, distance, num_cameras=8):
    """
    Add cameras to the MJCF file, evenly spaced on a sphere around the center.

    Parameters:
    - mjcf_file: str, path to the input MJCF file.
    - output_file: str, path to save the modified MJCF file.
    - center: numpy array of shape (3,), the center point.
    - distance: float, the distance from the center to each camera.
    - num_cameras: int, the number of cameras to add.
    """
    # Parse the MJCF file
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    # Find the <worldbody> element
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> element found in the MJCF file.")

    # Generate cameras
    cameras = generate_all_cameras(center, distance, num_cameras)

    # Add cameras to the <worldbody> as the first elements
    for i, camera in enumerate(reversed(cameras)):  # Reverse to maintain order
        position = " ".join(map(str, camera['position']))
        #quaternion = " ".join(map(str, camera['quaternion']))
        camera_element = ET.Element('camera', {
            'name': f'camera_{i}',
            'pos': position,
            #'quat': quaternion,
            'mode': 'targetbody',
            'target': root_name
        })
        worldbody.insert(0, camera_element)  # Insert at the beginning

    # Save the modified MJCF to a new file
    tree.write(output_file)
    print(f"MJCF file with cameras saved to {output_file}")

def get_robot_bounding_box(model, data, mjcf_file):
    # Get the root body ID (first body in the worldbody)
    root_body_id = model.body_rootid[1]  # Index 1 is the first body after the world body

    # Find all geoms associated with the root body and its children
    geom_ids = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        if body_id == root_body_id or model.body_parentid[body_id] == root_body_id:
            geom_ids.append(geom_id)

    # Initialize min and max bounds
    min_bound = np.array([-np.inf, -np.inf, -np.inf])
    max_bound = np.array([np.inf, np.inf, np.inf])


    # Iterate over all geoms
    for geom_id in geom_ids:
        # Use mj_geomDistance to compute the farthest points
        fromto = np.zeros(6)
        dist = mujoco.mj_geomDistance(model, data, geom_ids[0], geom_id, 1000.0, fromto)

        # Update min and max bounds
        min_bound = np.minimum(min_bound, fromto[:3])
        max_bound = np.maximum(max_bound, fromto[3:])

    tree = ET.parse(mjcf_file)
    root = tree.getroot()
    root_body = root.find("worldbody").find("body")
    body_pos = np.array([float(x) for x in root_body.get("pos", "0 0 0").split()])
    body_quat = np.array([float(x) for x in root_body.get("quat", "1 0 0 0").split()])

    return {
        'min': min_bound + body_pos,
        'max': max_bound + body_pos
    }


def _get_robot_bounding_box(model, mjcf_file):
    """
    Compute the bounding box of the robot using positions, orientations, and geom_rbound.
    Uses XML parsing to extract body hierarchy, positions, and orientations.
    """
    # Parse the XML model to extract body hierarchy, positions, and orientations
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    # Initialize min and max bounds with infinity
    min_bound = np.array([np.inf, np.inf, np.inf])
    max_bound = np.array([-np.inf, -np.inf, -np.inf])

    # Helper function to recursively traverse the body hierarchy
    def traverse_body(body_element, parent_pos, parent_quat):
        nonlocal min_bound, max_bound

        # Get the body's position and orientation relative to its parent
        body_pos = np.array([float(x) for x in body_element.get("pos", "0 0 0").split()])
        body_quat = np.array([float(x) for x in body_element.get("quat", "1 0 0 0").split()])

        # Check if the body quaternion has a zero norm
        if np.linalg.norm(body_quat) == 0:
            body_quat = np.array([1, 0, 0, 0])  # Default identity quaternion

        # Compute the global position and orientation of the body
        global_pos = parent_pos + np.dot(transform.Rotation.from_quat(parent_quat).as_matrix(), body_pos)
        global_quat = transform.Rotation.from_quat(parent_quat) * transform.Rotation.from_quat(body_quat)

        # Debugging: Print global position and orientation
        print(f"Body: {body_element.get('name', 'unnamed')}")
        print(f"Global Position: {global_pos}")
        print(f"Global Quaternion: {global_quat.as_quat()}")

        # Iterate through all geoms in the body
        for geom_element in body_element.findall("geom"):
            # Get the geom's position and orientation relative to the body
            geom_pos = np.array([float(x) for x in geom_element.get("pos", "0 0 0").split()])
            geom_quat = np.array([float(x) for x in geom_element.get("quat", "1 0 0 0").split()])

            # Check if the geom quaternion has a zero norm
            if np.linalg.norm(geom_quat) == 0:
                geom_quat = np.array([1, 0, 0, 0])  # Default identity quaternion

            # Compute the global position and orientation of the geom
            geom_global_pos = global_pos + np.dot(global_quat.as_matrix(), geom_pos)
            geom_global_quat = global_quat * transform.Rotation.from_quat(geom_quat)

            # Get the geom's ID to access geom_rbound from MuJoCo
            geom_name = geom_element.get("name")
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

            # Skip if geom_id is invalid (e.g., infinite planes)
            if geom_id == -1:
                continue

            # Get the bounding sphere radius
            rbound = model.geom_rbound[geom_id]

            # Debugging: Print geom global position and rbound
            print(f"Geom: {geom_name}")
            print(f"Geom Global Position: {geom_global_pos}")
            print(f"Geom Rbound: {rbound}")

            # Update the bounding box
            min_bound = np.minimum(min_bound, geom_global_pos - rbound)
            max_bound = np.maximum(max_bound, geom_global_pos + rbound)

        # Recursively traverse child bodies
        for child_element in body_element.findall("body"):
            traverse_body(child_element, global_pos, global_quat.as_quat())

    # Start traversal from the root body
    root_body = root.find("worldbody").find("body")
    body_pos = np.array([float(x) for x in root_body.get("pos", "0 0 0").split()])
    body_quat = np.array([float(x) for x in root_body.get("quat", "1 0 0 0").split()])

    traverse_body(root_body, body_pos, body_quat)  # Root position at [0, 0, 1.4]

    return {
        'min': min_bound + body_pos,
        'max': max_bound + body_pos
    }

def preprocess(mjcf_file, camera_distance=10., num_cameras=8):

    # Example usage
    #mjcf_file = '../xmls/environments/cheetahs/cheetah_7_full.xml'
    output_file = get_temp_filepath()


    # Example usage:
    # Assuming `model` and `data` are your MuJoCo model and data objects
    #mjcf_model = mujoco.MjModel.from_xml_path(mjcf_file)
    #data = mujoco.MjData(mjcf_model)
    #bounding_box = get_robot_bounding_box(mjcf_model, data, mjcf_file)


    # Extract root position and bounding box
    root_position, root_name = extract_root_position_and_bounding_box(mjcf_file)

    file_no_objects = get_temp_filepath()
    remove_skylights_skyboxes_floors(mjcf_file, file_no_objects)

    # Define the center of the sphere (e.g., the center of the bounding box)
    #bbox = np.array([
    #    (bounding_box['min'][0] + bounding_box['max'][0]),
    #    (bounding_box['min'][1] + bounding_box['max'][1]),
    #    (bounding_box['min'][2] + bounding_box['max'][2])
    #])

    #distance = np.linalg.norm(bbox[0] - bbox[1]) + 1

    # Add cameras to the MJCF file
    #distance = 2.0  # Distance from the center to each camera
    add_cameras_to_mjcf(file_no_objects, output_file, root_position, root_name, camera_distance, num_cameras=num_cameras)
    return output_file, num_cameras + 6, root_position
