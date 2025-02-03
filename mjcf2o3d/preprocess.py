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

    # Find all <include> elements
    includes = root.findall(".//include")

    # Remove each <include> element
    for include in includes:
        parent = include.getparent()
        if parent is not None:
            parent.remove(include)

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

def remove_visual_noise(mjcf_file, output_file):
    # Parse the MJCF file
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

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
        'rgba': "0 0 0 1"

    })
    asset.append(flat_material)

    # Apply flat material to all geoms
    for geom in root.findall(".//geom"):
        #if geom.get("material") is None:
        geom.set("material", "flat_material")

    # Save the modified MJCF
    tree.write(output_file)
    print(f"Modified MJCF saved to {output_file}")


def identify_actuator(mjcf_file, output_file):
    # Parse the MJCF file
    import colorsys
    tree = etree.parse(mjcf_file)
    root = tree.getroot()

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

    # Sort by depth (shallowest first) to process parent bodies first
    actuator_info.sort(key=lambda x: x[0])

    genned_colours = [np.array([0, 0, 0, 255])]

    def get_random_colour(i, num_actuators):
        np.random.seed(i + num_actuators)

        for _ in range(1000):
            ret = np.random.randint(0, 255, size=(4))
            ret[-1] = 255  # Ensure full opacity

            valid = True
            for genned_colour in genned_colours:
                if np.all(np.abs(ret[:-1] - genned_colour[:-1]) < 2):  # Check R, G, B
                    valid = False
                    break

            if valid:
                genned_colours.append(ret)
                return tuple(ret.astype(float) / 255)
        raise AssertionError("Could not find a valid colour")

    # Generate distinct colors for each actuator
    num_actuators = len(actuator_info)
    actuator_colors = {}
    for i, (depth, actuator_name, joint_name, geoms) in enumerate(actuator_info):
        hue = i / num_actuators  # Vary hue from 0 to 1
        #hue  = hue / 2
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Full saturation and value
        #colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Full saturation and value
        rgba = (r, g, b, 1.0)  # Alpha set to 1.0
        #rgba = (0.5, 0.0, 1.0, 1.0)
        #if i < 3:
        #    rgba = (1.0, 0.0, 0.75, 1.0)
        rgba = get_random_colour(i, num_actuators)
        actuator_colors[actuator_name] = rgba

        # Apply color to all geoms in this actuator's hierarchy
        rgba_str = ' '.join(f"{c:.3f}" for c in rgba)
        for geom in geoms:
            geom.set('rgba', rgba_str)

    # Save the modified MJCF to a new file
    tree.write(output_file)
    print(f"Actuator-colored MJCF saved to {output_file}")
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

def preprocess(mjcf_file, num_cameras=8, isolate_actuators=False):
    root_position, root_name = extract_root_position_and_name(mjcf_file)

    # Compute bounding box
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    bounding_box = _get_robot_bounding_box(model, mjcf_file)
    size = bounding_box['max'] - bounding_box['min']
    diagonal = np.linalg.norm(size)
    camera_distance = diagonal * 3.0

    file_no_planes= get_temp_filepath()
    remove_skylights_skyboxes_floors(mjcf_file, file_no_planes)

    workfile = file_no_planes
    actuator_colors = {}
    if isolate_actuators:
        file_no_objects = get_temp_filepath()
        remove_visual_noise(file_no_planes, file_no_objects)

        identified_actuators = get_temp_filepath()
        actuator_colors = identify_actuator(file_no_objects, identified_actuators)
        workfile = identified_actuators

    output_file = get_temp_filepath()
    add_cameras_to_mjcf(workfile, output_file, root_position, root_name, camera_distance, isolate_actuators=isolate_actuators, num_cameras=num_cameras)
    return output_file, num_cameras + 6, camera_distance, root_position, actuator_colors
