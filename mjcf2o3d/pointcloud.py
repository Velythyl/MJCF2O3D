from tqdm import tqdm
import open3d as o3d
import numpy as np


def pc_to_pcd(p, c):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

def pc_mask(p, c, indices_to_remove):
    # Remove the points marked for removal
    mask = np.ones(p.shape[0])
    for i in indices_to_remove:
        mask[i] = 0
    mask = mask.astype(bool)

    return p[mask], c[mask]

def pc_cleanup_distance_outliers(p, c):
    print("Cleaning up point cloud for floating points")
    pcd = pc_to_pcd(p, c)
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

    p, c = pc_mask(p, c, outlier_indices)

    return p, c, (mean_distance, std_dev_distance)


def pc_cleanup_knn_colours(p, c):
    # todo rm unused
    pcd = pc_to_pcd(p, c)

    print("Cleaning up point cloud for in-geometry bad data")

    # Build a KDTree for nearest neighbor search

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # List to store indices of points to be removed
    N_KNN = 3
    indices_to_remove = []
    # Iterate through each point in the point cloud
    for i in tqdm(range(len(pcd.points))):
        # Find the 10 nearest neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], N_KNN)

        # Get the color of the current point
        current_color = np.asarray(pcd.colors)[i]

        # Check the colors of the neighbors
        same_color_count = np.all(current_color == np.asarray(pcd.colors)[idx], axis=1).sum()

        # If fewer than a certain number of neighbors have the same color, mark for removal
        if same_color_count < N_KNN:  # You can adjust this threshold
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

