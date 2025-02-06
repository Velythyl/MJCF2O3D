import argparse
import gc
import json
import multiprocessing
import os
import shutil
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from mjcf2o3d.file_utils import get_temp_filename, get_temp_filepath, save_json
from mjcf2o3d.pointcloud import pc_to_pcd, pc_cleanup_distance_outliers
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

def main_isolate(mjcf_file, fullscan_gif_file):
    num_cameras = 32
    intermediate_file, num_cameras, camera_distance, root_position = preprocess(mjcf_file,
                                                                                                 num_cameras=num_cameras,
                                                                                                 isolate_actuators=True)

    actuator_names = get_actuator_names(intermediate_file)

    def is_closer_to_white(pcd_colours):
        return np.all(pcd_colours != np.array([0,0,0]), axis=1)


    _, actuator_colors = handle_actuator(intermediate_file, [actuator_names[0]])

    rebuild_pcd_points = []
    rebuild_pcd_colours = []
    gif_files = [fullscan_gif_file]

    def handle_actuator_list(actuator_name_list, replace_by_colour, rest_of_body="t", kept_actuators="w"):
        file_with_isolated_actuator, _ = handle_actuator(intermediate_file, actuator_name_list, rest_of_body, "t", kept_actuators)

        pcd, giffile = scanfile(mjcf_file, file_with_isolated_actuator, root_position, num_cameras, camera_distance)

        pcd_points = np.asarray(pcd.points)
        pcd_points = pcd_points + np.random.normal(loc=0.0, scale=0.001, size=pcd_points.shape)
        pcd_colours = np.asarray(pcd.colors)

        isolated_actuator_colour_mask = is_closer_to_white(pcd_colours)

        actuator_specific_colour = np.full_like(pcd_colours[isolated_actuator_colour_mask],
                                                replace_by_colour)

        rebuild_pcd_points.append(pcd_points[isolated_actuator_colour_mask])
        rebuild_pcd_colours.append(actuator_specific_colour)
        gif_files.append(giffile)

    for actuator_name in actuator_names:
        handle_actuator_list([actuator_name], actuator_colors[actuator_name])
    # GRABS THE REST OF THE BODIES
    handle_actuator_list(actuator_names, (0,0,0), "w", "t")
    actuator_names.append(">NO_ACTUATORS<")
    actuator_colors[actuator_names[-1]] = (0,0,0)

    def handle_gifs():
        # Open images and store them in a list
        frames = [Image.open(gif) for gif in gif_files]
        # Save as animated GIF
        image_path = f"{get_temp_filepath('.gif')}"
        frames[0].save(image_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        return image_path

    new_image_path = handle_gifs()

    cleaned_pcd_points, cleaned_pcd_colours = [], []
    for p,c in zip(rebuild_pcd_points, rebuild_pcd_colours):
        p, c, _ = pc_cleanup_distance_outliers(p, c)
        cleaned_pcd_points.append(p)
        cleaned_pcd_colours.append(c)

    pcd_points, pcd_colours = np.concatenate(cleaned_pcd_points), np.concatenate(cleaned_pcd_colours)
    _, _, (_, std_distances) = pc_cleanup_distance_outliers(pcd_points, pcd_colours)
    pcd_points = pcd_points + np.random.normal(loc=0.0, scale=std_distances / 100, size=pcd_points.shape)

    info_json = {">FULL<": {"color": "all", "pcd_points": pcd_points.tolist(), "pcd_colors": pcd_colours.tolist(), "pcd_indices": np.arange(pcd_points.shape[0]).tolist()}}
    for actuator_name, color in actuator_colors.items():
        mask = np.all(pcd_colours == color, axis=1)
        assert mask.sum() > 0
        info_json[actuator_name] = {
            "color": list(color),
            "pcd_points": pcd_points[mask].tolist(),
            "pcd_indices": np.arange(pcd_points.shape[0])[mask].tolist(),
            "pcd_colors": pcd_colours[mask].tolist()
        }

    return pc_to_pcd(pcd_points, pcd_colours), info_json, new_image_path



def mass_main(mjcf_tree, do_visualize, isolate_actuators, log_file=None, refresh=False):
    """
    Traverse a directory tree, find all XML files, and call the `main` function for each XML file.

    Args:
        mjcf_tree (str): The root directory of the XML file tree.
        do_visualize (bool): Whether to visualize the point cloud.
        isolate_actuators (bool): Whether to isolate actuators in the point cloud.
    """
    # Collect and filter XML file paths
    xml_files = []
    for root, _, files in os.walk(mjcf_tree):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)

                if file.startswith("mjcf2o3d"):
                    os.remove(xml_path)
                    continue

                pcd_path = xml_path.replace(".xml", "-parsed.json")
                if refresh or (not os.path.exists(pcd_path)):  # Skip if already processed
                    xml_files.append(xml_path)

    if log_file is None:
        log_file = get_temp_filepath(ext=".txt")

    # Redirect stdout and stderr to log file
    print(f"Write mass process log to {log_file}")
    print()
    log_file = open(log_file, "w")

    # Process files with tqdm progress bar
    pbar = tqdm(xml_files, desc="Processing XML files")
    for mjcf_file in pbar:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = log_file, log_file

        print(f"\n\n>>> NOW DOING {mjcf_file}\n\n")

        outfile = mjcf_file.replace(".xml", "-parsed.pcd")
        pbar.set_postfix_str(mjcf_file)
        try:
            main(mjcf_file, outfile, do_visualize, isolate_actuators)
        except Exception as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            raise e

        print(f"\n\n>>> DONE DOING {mjcf_file}\n\n")

        sys.stdout, sys.stderr = old_stdout, old_stderr

    log_file.close()
    print(f"Wrote log to {log_file}")

def main(mjcf_file, outfile, do_visualize, isolate_actuators):
    pcd, temp_image_path = main_no_isolate(mjcf_file)
    json_format = {"color": "all", "pcd_points": np.asarray(pcd.points).tolist(), "pcd_colors": np.asarray(pcd.colors).tolist(), "pcd_indices": np.arange(np.asarray(pcd.points).shape[0]).tolist()}
    json_format = {">ORIGINAL XML<": json_format}
    if isolate_actuators:
        pcd, new_json_format, temp_image_path = main_isolate(mjcf_file, temp_image_path)
        json_format.update(new_json_format)

    json_outfile = outfile.split(".pcd")[0]+".json"
    save_json(json_format, json_outfile)
    print(f"Saved JSON using GZIP to {outfile}")

    # Save the image if imagepath is provided
    image_out =  outfile.split(".pcd")[0]+".gif"
    shutil.move(temp_image_path, image_out)
    print(f"Saved image to {image_out}")

    # Visualize the point cloud if requested
    if do_visualize:
        visualize(pcd)


if __name__ == '__main__':
    #mp_main("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100", isolate_actuators=True)
    #exit()

    mass_main("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100", do_visualize=False, isolate_actuators=True)
    exit()

    with open("./pointcloud.json", "r") as f:
        json_loaded = json.load(f)
    key1 = list(json_loaded.keys())[2]
    p, c = np.asarray(json_loaded[key1]["pcd_points"]), np.asarray(json_loaded[key1]["pcd_colors"])
    visualize(pc_to_pcd(p,c))
    #key2 = json_loaded[list(json_loaded.keys())[1]]

    #mass_main("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100", do_visualize=True, isolate_actuators=True)
    exit()

    #main("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/kinematics/xml/mvt-5506-12-6-17-12-20-06_limb_params_3.xml", "./pointcloud.pcd", True, True)
    #exit()

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