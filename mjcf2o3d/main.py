import argparse
import gc
import json
import os
import shutil

import numpy as np
from tqdm import tqdm

from mjcf2o3d.file_utils import get_temp_filename
from mjcf2o3d.preprocess import preprocess, get_actuator_names, handle_actuator
from mjcf2o3d.scanner.dm_control_scan import scan

import open3d as o3d

def visualize(pcd):

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()

    view_control = visualizer.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass

def scanfile(mjcf_file, workfile, root_position, num_cameras, camera_distance):
    intermediate_in_folder = f"/".join(mjcf_file.split("/")[:-1]) + f"/{get_temp_filename()}.xml"

    shutil.copy(workfile, intermediate_in_folder)

    pcd, temp_image_path = scan(intermediate_in_folder, root_position, num_cameras=num_cameras,
                                max_distance=camera_distance)
    gc.collect()
    os.remove(intermediate_in_folder)
    return pcd, temp_image_path


def main_no_isolate(mjcf_file):
    num_cameras = 32
    intermediate_file, num_cameras, camera_distance, root_position = preprocess(mjcf_file,
                                                                                                 num_cameras=num_cameras,
                                                                                                 isolate_actuators=False)
    return scanfile(mjcf_file, intermediate_file, root_position, num_cameras, camera_distance)

def main_isolate(mjcf_file):
    num_cameras = 32
    intermediate_file, num_cameras, camera_distance, root_position = preprocess(mjcf_file,
                                                                                                 num_cameras=num_cameras,
                                                                                                 isolate_actuators=True)

    actuator_names = get_actuator_names(mjcf_file)

    def is_closer_to_white(pcd_colours):
        return np.all(pcd_colours != np.array([0,0,0]), axis=1)


    _, actuator_colors = handle_actuator(intermediate_file, actuator_names[0])

    rebuild_pcd_points = []
    rebuild_pcd_colours = []

    def handle_actuator_list(actuator_name_list, replace_by_colour, flip_mask):
        file_with_isolated_actuator, _ = handle_actuator(intermediate_file, actuator_name_list)

        pcd, _ = scanfile(mjcf_file, file_with_isolated_actuator, root_position, num_cameras, camera_distance)

        pcd_points = np.asarray(pcd.points)
        pcd_colours = np.asarray(pcd.colors)

        isolated_actuator_colour_mask = is_closer_to_white(pcd_colours)
        if flip_mask:
            isolated_actuator_colour_mask = np.logical_not(isolated_actuator_colour_mask)

        actuator_specific_colour = np.full_like(pcd_colours[isolated_actuator_colour_mask],
                                                replace_by_colour)

        rebuild_pcd_points.append(pcd_points[isolated_actuator_colour_mask])
        rebuild_pcd_colours.append(actuator_specific_colour)

    rebuild_pcd_points = []
    rebuild_pcd_colours = []
    for actuator_name in actuator_names:
        handle_actuator_list([actuator_name], actuator_colors[actuator_name], False)
    # GRABS THE REST OF THE BODIES
    handle_actuator_list(actuator_names, (0,0,0), True)
    pcd_points = np.concatenate(rebuild_pcd_points)
    pcd_colours = np.concatenate(rebuild_pcd_colours)

    def pc_to_pcd(p, c):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        pcd.colors = o3d.utility.Vector3dVector(c)
        return pcd

    print("Cleaning up point cloud for in-geometry bad data")

    # Build a KDTree for nearest neighbor search
    pcd = pc_to_pcd(pcd_points, pcd_colours)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # List to store indices of points to be removed
    indices_to_remove = []
    # Iterate through each point in the point cloud
    for i in tqdm(range(len(pcd.points))):
        # Find the 10 nearest neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)

        # Get the color of the current point
        current_color = np.asarray(pcd.colors)[i]

        # Check the colors of the neighbors
        same_color_count = 0
        for neighbor_idx in idx:
            neighbor_color = np.asarray(pcd.colors)[neighbor_idx]
            if np.allclose(neighbor_color, current_color, atol=0.1):  # You can adjust the tolerance
                same_color_count += 1

        # If fewer than a certain number of neighbors have the same color, mark for removal
        if same_color_count < 5:  # You can adjust this threshold
            indices_to_remove.append(i)

    def pc_mask(p, c, indices_to_remove):
        # Remove the points marked for removal
        mask = np.ones(pcd_points.shape[0])
        for i in indices_to_remove:
            mask[i] = 0
        mask = mask.astype(bool)

        return p[mask], c[mask]

    # Remove the points marked for removal
    pcd_points, pcd_colours = pc_mask(pcd_points, pcd_colours, indices_to_remove)

    print("Cleaning up point cloud for floating points")
    pcd = pc_to_pcd(pcd_points, pcd_colours)
    # Build a KDTree for nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Parameters
    k = 10  # Number of neighbors to consider
    std_dev_threshold = 10.0  # Threshold for outlier detection (in terms of standard deviations)

    # List to store the average distance of each point to its k nearest neighbors
    avg_distances = []

    # Iterate through each point in the point cloud
    for i in tqdm(range(len(pcd.points))):
        # Find the k nearest neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)

        # Compute the average distance to the neighbors
        distances = np.linalg.norm(np.asarray(pcd.points)[idx] - np.asarray(pcd.points)[i], axis=1)
        avg_distance = np.mean(distances)
        avg_distances.append(avg_distance)

    # Convert to a numpy array for easier calculations
    avg_distances = np.array(avg_distances)

    # Compute the mean and standard deviation of the average distances
    mean_distance = np.mean(avg_distances)
    std_dev_distance = np.std(avg_distances)

    # Identify outliers (points with average distance > mean + std_dev_threshold * std_dev)
    outlier_indices = np.where(avg_distances > mean_distance + std_dev_threshold * std_dev_distance)[0]

    pcd_points, pcd_colours = pc_mask(pcd_points, pcd_colours, outlier_indices)

    pcd = pc_to_pcd(pcd_points, pcd_colours)

    return pcd, actuator_colors


def mass_main(mjcf_tree, do_visualize, isolate_actuators):
    """
    Traverse a directory tree, find all XML files, and call the `main` function for each XML file.

    Args:
        mjcf_tree (str): The root directory of the XML file tree.
        do_visualize (bool): Whether to visualize the point cloud.
        isolate_actuators (bool): Whether to isolate actuators in the point cloud.
    """
    for root, _, files in os.walk(mjcf_tree):
        for file in files:
            if file.endswith(".xml"):
                mjcf_file = os.path.join(root, file)
                outfile = mjcf_file.replace(".xml", "-parsed.pcd")
                main(mjcf_file, outfile, do_visualize, isolate_actuators)

def main(mjcf_file, outfile, do_visualize, isolate_actuators):
    pcd, temp_image_path = main_no_isolate(mjcf_file)
    if isolate_actuators:
        pcd, actuator_colors = main_isolate(mjcf_file)


    # Save the point cloud to the specified outfile
    o3d.io.write_point_cloud(outfile, pcd)
    print(f"Saved PCD to {outfile}")

    if isolate_actuators:
        json_outfile = outfile.split(".pcd")[0]+".json"
        jsonstring = json.dumps(actuator_colors).replace(", ", ",\n ")
        with open(json_outfile, "w") as f:
            f.write(jsonstring)
        print(f"Saved actuator colours to {outfile}")

    # Save the image if imagepath is provided
    image_out =  outfile.split(".pcd")[0]+".gif"
    shutil.move(temp_image_path, image_out)
    print(f"Saved image to {image_out}")

    # Visualize the point cloud if requested
    if do_visualize:
        visualize(pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process MJCF file to generate point cloud and optionally save an image.")

    # "/home/charlie/Desktop/MJCFConvert/mjcf2o3d/xmls/environments/cheetahs/cheetah_3_back.xml"
    parser.add_argument('infile', type=str, help="Path to the input MJCF file.")
    parser.add_argument('--outfile', type=str, default="./pointcloud.pcd",
                        help="Path to save the output point cloud file.")
    parser.add_argument('--isolateactuators', action='store_true', help="Colour geoms by their actuators and saves the actuator colour dict.")
    parser.add_argument('--visualize', action='store_true', help="Visualize the point cloud after generation.")

    args = parser.parse_args()

    main(args.infile, args.outfile, args.visualize, args.isolateactuators)