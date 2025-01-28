import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib
matplotlib.use("tkagg")


def generate_cameras_on_axes(center, distance):
    """
    Generate cameras along the standard axes (X, Y, Z) at a given distance from the center.

    Parameters:
    - center: numpy array of shape (3,), the center point.
    - distance: float, the distance from the center to each camera.

    Returns:
    - cameras: list of dictionaries, each containing 'position' and 'quaternion' keys.
    """
    cameras = []

    # Define positions along the standard axes
    axis_positions = np.array([
        [distance, 0, 0],  # Right (X+)
        [-distance, 0, 0],  # Left (X-)
        [0, distance, 0],  # Front (Y+)
        [0, -distance, 0],  # Back (Y-)
        [0, 0, distance],  # Up (Z+)
        [0, 0, -distance]   # Down (Z-)
    ]) + center

    # Calculate quaternions for each camera to point towards the center
    for position in axis_positions:
        cameras.append({
            'position': position
        })

    return cameras

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
        cameras.append({
            'position': position,
        })

    return cameras


def generate_all_cameras(center, distance, num_extra_cameras=8):
    axes = generate_cameras_on_axes(center, distance)
    sphere = generate_cameras_on_sphere(center, distance, num_cameras=num_extra_cameras)

    return axes + sphere


def plot_cameras(cameras, center):
    """
    Plot the cameras and their orientations in 3D.

    Parameters:
    - cameras: list of dictionaries, each containing 'position' and 'quaternion' keys.
    - center: numpy array of shape (3,), the center point.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract camera positions
    positions = np.array([camera['position'] for camera in cameras])

    # Plot the center point
    ax.scatter(*center, color='red', s=100, label='Center')

    # Plot the camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='blue', label='Cameras')

    # Calculate and plot the direction vectors
    vectors = []
    for camera in cameras:
        position = camera['position']
        quaternion = camera['quaternion']

        # Convert quaternion to a rotation matrix
        rotation = R.from_quat(quaternion)

        # Define the default forward direction (assumed to be along -z)
        forward = np.array([0, 0, -1])

        # Rotate the forward direction using the quaternion
        direction = rotation.apply(forward)

        # Scale the direction vector for visualization
        direction *= 2  # Arbitrary scaling for better visualization

        # Add the vector to the list
        vectors.append([position, position + direction])

    # Convert vectors to a Line3DCollection for efficient plotting
    lines = Line3DCollection(vectors, colors='green', label='Camera Directions')
    ax.add_collection(lines)

    # Set plot limits
    max_range = np.max(np.abs(positions - center)) * 1.5
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title('Camera Positions and Orientations on a Sphere')
    plt.show()


if __name__ == "__main__":
    # Example usage
    center = np.array([0, 0, 1.4])
    distance = 5.0
    num_cameras = 16  # You can change this to any number of cameras
    cameras = generate_all_cameras(center, distance, num_cameras)

    # Plot the cameras
    plot_cameras(cameras, center)

    def listitem2mujostr(item):
        item = round(item,2)
        return str(item)


    strings = []
    for i, dico in enumerate(cameras):
        pos = dico["position"]
        quat = dico["quaternion"]


        string = f"""<camera mode="trackcom" name="face_{i}" pos="{' '.join(map(listitem2mujostr,pos))}" quat="{' '.join(map(listitem2mujostr,quat))}" />"""
        strings.append(string)

    print("\n".join(strings))