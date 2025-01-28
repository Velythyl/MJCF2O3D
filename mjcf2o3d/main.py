import argparse
import os
import shutil

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


def main(mjcf_file, num_cameras, camera_distance, outfile, imagepath, do_visualize):
    intermediate_file, num_cameras, root_position = preprocess(mjcf_file, camera_distance=camera_distance, num_cameras=num_cameras)

    intermediate_in_folder = f"/".join(mjcf_file.split("/")[:-1])  + f"/{get_temp_filename()}.xml"

    shutil.copy(intermediate_file, intermediate_in_folder)

    pcd, temp_image_path = scan(intermediate_in_folder, root_position, num_cameras=num_cameras, max_distance=camera_distance)

    os.remove(intermediate_in_folder)

    # Save the point cloud to the specified outfile
    o3d.io.write_point_cloud(outfile, pcd)
    print(f"Saved PCD to {outfile}")

    # Save the image if imagepath is provided
    if imagepath:
        shutil.copy(temp_image_path, imagepath)
        print(f"Saved image to {imagepath}")

    # Visualize the point cloud if requested
    if do_visualize:
        visualize(pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process MJCF file to generate point cloud and optionally save an image.")

    parser.add_argument('infile', type=str, help="Path to the input MJCF file.")
    parser.add_argument('--outfile', type=str, default="./pointcloud.pcd",
                        help="Path to save the output point cloud file.")
    parser.add_argument('--num_cameras', type=int, default=16, help="Number of cameras to use for scanning.")
    parser.add_argument('--camera_distance', type=float, default=5.0, help="Distance of the cameras from the object.")
    parser.add_argument('--imagepath', type=str, default=None,
                        help="Path to save the rendered image. If not provided, the image will not be saved.")
    parser.add_argument('--visualize', action='store_true', help="Visualize the point cloud after generation.")

    args = parser.parse_args()

    main(args.infile, args.num_cameras, args.camera_distance, args.outfile, args.imagepath, args.visualize)