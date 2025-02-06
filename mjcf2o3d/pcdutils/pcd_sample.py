import numpy as np

from mjcf2o3d.file_utils import load_json


def is_single_color(c):
    return (c[1:] == c[:-1]).all()


def pc_downsample(p, c, num_keep):
    assert isinstance(num_keep, int)

    indices = np.arange(p.shape[0])

    if is_single_color(c):
        # in this case, we have a single-ID point cloud and we can directly resample
        selected_indices = np.random.choice(indices, size=num_keep, replace=False, p=None)

    else:
        unique_colors, num_each_color = np.unique(c, axis=0, return_counts=True)
        color_probabilities = num_each_color / num_each_color.sum()  # Normalize to get probabilities

        # Determine how many points to keep per color
        num_keep_per_color = np.round(color_probabilities * num_keep).astype(int)

        # Ensure the total number of selected points is exactly num_keep
        num_keep_per_color[np.argmax(num_keep_per_color)] += num_keep - num_keep_per_color.sum()

        selected_indices = []
        for color, keep_count in zip(unique_colors, num_keep_per_color):
            color_indices = indices[(c == color).all(axis=1)]
            selected_indices.extend(np.random.choice(color_indices, size=keep_count, replace=False))

        selected_indices = np.array(selected_indices)

    return p[selected_indices], c[selected_indices]

def pc_to_pcd(p, c):
    # todo merge with pointcloud.py
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

EPSILON = 0.0001
def pc_get_rescale(p,c):
    translate = 0 - p.min(axis=0)
    translated_p = p + translate + EPSILON
    assert (translated_p.min(axis=0) >= 0).all()

    downscale_factor = 1 / translated_p.max()

    return (lambda p,c: ((p + translate + EPSILON) * downscale_factor,c))

def pc_to_voxel(p,c, voxel_grid_size=16, rescale_func=None):

    if rescale_func is None:
        rescale_func = pc_get_rescale(p,c)

    p,c = rescale_func(p,c)

    assert (p.min(axis=0) >= 0).all()
    assert (p.max(axis=0) <= 1).all()

    bounds = (p.min(axis=0), p.min(axis=0))
    assert (bounds[0] >= 0).any()
    assert (bounds[1] <= 1).any()

    pcd = pc_to_pcd(p,c)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1/voxel_grid_size)
    o3d.visualization.draw_geometries([voxel_grid])


if __name__ == "__main__":
    import json
    path = "/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/test/xml/floor-5506-6-3-01-15-20-20-parsed.json"

    json_loaded = load_json(path)
    key1 = list(json_loaded.keys())[2]
    mask = np.all(json_loaded[">FULL<"]["pcd_colors"]  ==json_loaded[key1]["color"], axis=1)
    p = (json_loaded[">FULL<"]["pcd_points"])[mask]
    c = (json_loaded[">FULL<"]["pcd_colors"])[mask]
    p, c = (p), (c)

    import open3d as o3d

    pc_to_voxel(p,c)
    exit()

    p, c = pc_downsample(p,c, 8192)

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

    visualize(pc_to_pcd(p, c))

    with open("temp.npy", "w") as f:
        np.save("temp.npy", p)