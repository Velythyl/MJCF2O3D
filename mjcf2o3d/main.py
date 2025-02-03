import argparse
import gc
import json
import os
import shutil

import numpy as np

from mjcf2o3d.file_utils import get_temp_filename
from mjcf2o3d.preprocess import preprocess
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


def main(mjcf_file, outfile, do_visualize, isolate_actuators):
    num_cameras = 32
    intermediate_file, num_cameras, camera_distance, root_position, actuator_colors = preprocess(mjcf_file, num_cameras=num_cameras, isolate_actuators=isolate_actuators)

    intermediate_in_folder = f"/".join(mjcf_file.split("/")[:-1])  + f"/{get_temp_filename()}.xml"

    shutil.copy(intermediate_file, intermediate_in_folder)

    pcd, temp_image_path = scan(intermediate_in_folder, root_position, num_cameras=num_cameras, max_distance=camera_distance)
    gc.collect()
    os.remove(intermediate_in_folder)

    pcd_points = np.asarray(pcd.points)
    pcd_colours = np.asarray(pcd.colors)

    pcd_colours = (pcd_colours * 255).astype(int)
    _actuator_colors = {k: (np.array(v)[:-1] * 255).astype(int) for k,v in actuator_colors.items()}

    for a_color in _actuator_colors.values():
        print(np.all(pcd_colours == a_color, axis=1).sum())

    # np.all(pcd_colours == list(actuator_colors.values())[4], axis=1).sum()


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