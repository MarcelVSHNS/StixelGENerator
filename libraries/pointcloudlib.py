import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple
from enum import Enum, auto
from names import point_dtype


def remove_ground(points: np.array, distance_threshold: float = 0.18, ransac_n: int = 3, num_iterations: int = 1000) -> np.array:
    """
    Removes the ground from a point cloud and retains the associated projection data.
    This function identifies and removes the points that belong to the ground plane
    using the RANSAC algorithm. It then extracts the projection data from the outlier
    points which are not part of the ground.
    Args:
        points: The input array containing point cloud data and projection data.
                              Expected shape is (N, 5) where N is the number of points.
                              The first three columns are x, y, z coordinates of the point cloud.
                              The last two columns proj_x, proj_y are the projection data.
        distance_threshold: The maximum distance a point can be from the plane model
                                  to be considered an inlier.
        ransac_n: The number of points to sample for estimating the plane.
        num_iterations: The number of iterations to run the RANSAC algorithm.
    Returns:
        combined_data_without_ground: An array of shape (M, 5), where M is the number of
                                                    points not belonging to the ground. The first three
                                                    columns are the x, y, z coordinates, and the last two
                                                    columns are the associated projection data.
    """
    # Convert the input array to a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # Use RANSAC to segment the ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    # Select points that are not part of the ground plane
    pcd_without_ground = pcd.select_by_index(inliers, invert=True)
    # Create a boolean mask for inliers
    inlier_mask = np.ones(len(points), dtype=bool)
    inlier_mask[inliers] = False
    outlier_indices = np.arange(len(points))[inlier_mask]
    # Extract the projection data for the outliers
    projection_data = points[outlier_indices, 3:]
    # Convert the point cloud without ground to a NumPy array
    pcd_without_ground_np = np.asarray(pcd_without_ground.points)
    # Merge the point cloud without ground with the projection data
    combined_data_without_ground = np.hstack((pcd_without_ground_np, projection_data))
    return combined_data_without_ground     # np.array(combined_data_without_ground, dtype=point_dtype)


def remove_line_of_sight(points: np.array, camera_position_xyz: np.array) -> np.array:
    # Manually extract x, y, z data from the structured array
    xyz = np.array([points['x'], points['y'], points['z']]).T.astype(np.float64)
    # Verify the shape and dtype of the xyz array
    if xyz.shape[1] != 3 or xyz.dtype != np.float64:
        raise ValueError("The xyz array must be of shape (n_points, 3) and dtype np.float64")
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Führe die Hidden Point Removal-Funktion aus
    camera = [camera_position_xyz[0], camera_position_xyz[1], camera_position_xyz[2]]
    radius = 100  # Radius der Sichtbarkeit um die Kameraposition. Muss eventuell angepasst werden
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    # Konvertiere das Ergebnis zurück in ein strukturiertes NumPy-Array
    # Indizes der sichtbaren Punkte
    visible_indices = np.asarray(pt_map)
    # Erstellen eines neuen strukturierten Arrays für die sichtbaren Punkte
    visible_points_structured = np.zeros(len(visible_indices), dtype=points.dtype)
    # Zuweisung der sichtbaren Punkte und ihrer proj_x, proj_y Werte
    for name in points.dtype.names:
        visible_points_structured[name] = points[name][visible_indices]
    return visible_points_structured


def remove_far_points(points: np.array, range_threshold: float = 60.0) -> np.array:
    # Calculate the distance (range) from x and y for each point
    ranges = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    # Filter out points where the distance is greater than the threshold
    return points[ranges <= range_threshold]


def group_points_by_angle(points: np.array, eps: float = 0.00092, min_samples: int = 1) -> List[np.array]:
    """
    Groups points based on their azimuth angle and returns a list of arrays,
    each containing the points (x, y, z, proj_x, proj_y) of the same angle.

    :param points: A numpy array of points (x, y, z, proj_x, proj_y).
    :param eps: The maximum distance between two points to be considered in the same cluster.
    :param min_samples: The number of points required in a neighborhood for a point to be considered a core point.
    :return: A list of numpy arrays, each containing points of an angle cluster.
    """
    # Compute the azimuth angle for each point
    points = points[:, :3]
    azimuth_angles = np.arctan2(points[:, 1], points[:, 0])
    # Normalize angles between 0 and 2*pi
    # azimuth_angles = np.mod(azimuth_angles, 2 * np.pi)
    # Perform DBSCAN clustering based on azimuth angles only
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(azimuth_angles.reshape(-1, 1))
    # Obtain cluster labels
    labels = db.labels_
    # Group points based on labels and compute average angles
    unique_labels, unique_indices = np.unique(labels, return_index=True)
    # Skip outliers if present
    valid_indices = unique_labels != -1
    unique_labels = unique_labels[valid_indices]
    unique_indices = unique_indices[valid_indices]
    grouped_points = [points[labels == label] for label in unique_labels]
    angle_means = [azimuth_angles[labels == label].mean() for label in unique_labels]
    # Sort clusters and average angles by the angles
    angle_means, grouped_points = zip(*sorted(zip(angle_means, grouped_points)))
    return list(grouped_points)         #, list(angle_means)


def _euclidean_distance_with_raising_eps():
    pass


class PositionClass(Enum):
    BOTTOM = auto()
    TOP = auto()


class Stixel:
    def __init__(self, point: np.array, position_class: PositionClass, floor_offset: int = 0):
        self.point: np.array = point
        self.point['proj_y'] += 32
        self.position_class: PositionClass = position_class
        # If stixel is a bottom stixel...
        if self.position_class == PositionClass.BOTTOM:
            # ... apply the floor offset (in px) to proj_y
            self.point['proj_y'] += floor_offset


class Cluster:
    def __init__(self, points: np.array):
        self.points: np.array = points            # Shape: x, y, z, proj_x, proj_y
        self.mean_range: float = self.calculate_mean_range()
        self.stixels: List[Stixel] = []
        self.sort_points_by_z()

    def calculate_mean_range(self) -> float:
        distances: List[float] = [np.sqrt(point[0]**2 + point[1]**2) for point in self.points]
        return np.mean(distances)

    def sort_points_by_z(self):
        # Sort points by ascending z
        self.points = sorted(self.points, key=lambda point: point[2])


class Scanline:
    def __init__(self, points: np.array):
        self.points: np.array = points.view(dtype=point_dtype).reshape(-1)
        self.objects: List[Cluster] = []
        self._cluster_objects()
        self._determine_stixel()

    def _cluster_objects(self, eps: float = 1.0, min_samples: int = 1, metric: str = 'euclidean'):
        # Compute the radial distance r
        r_values = np.sqrt(self.points['x'] ** 2 + self.points['y'] ** 2)
        # Sort points by r for clustering
        sorted_indices = np.argsort(r_values)
        sorted_r = r_values[sorted_indices]
        sorted_z = self.points['z'][sorted_indices]
        self.points = self.points[sorted_indices]
        # Prepare the data for DBSCAN
        db_data = np.column_stack((sorted_r, sorted_z))
        # Check if enough data points are present for clustering
        if len(db_data) > 1:
            # Apply the DBSCAN clustering algorithm
            db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(db_data)
            labels = db.labels_
        else:
            # Treat the single point as its own cluster
            labels = np.array([0])
        # Identify stixels by cluster
        for label in np.unique(labels):
            if label == -1:
                continue  # Skip outliers
            # Create a Cluster object for each group of points sharing the same label
            cluster_points = self.points[labels == label]
            self.objects.append(Cluster(cluster_points))
        # Sort the list of Cluster objects by their mean_range
        self.objects = sorted(self.objects, key=lambda cluster: cluster.mean_range)

    def _determine_stixel(self, distance_threshold: float = 10.0, x_threshold: float = 0.2, extended_stixel: bool = True):
        previous_cluster = None
        last_top_stixel_x = None  # Speichert die x-Position des letzten Top-Stixels

        for cluster in self.objects:
            # Add the bottom point of the first cluster or if the distance to the previous cluster is larger than the threshold
            if previous_cluster is None or (cluster.mean_range - previous_cluster.mean_range) > distance_threshold:
                bottom_point = cluster.points[0]  # First point after sorting by z (lowest z)
                bottom_stixel = Stixel(point=bottom_point, position_class=PositionClass.BOTTOM)  # 0 for Bottom
                cluster.stixels.append(bottom_stixel)
                last_top_stixel_x = bottom_point[0]  # Aktualisiere die x-Position für die Top-Stixel-Logik
            if extended_stixel:
                # Iterate through each point in the cluster to check the x_threshold condition
                for point in cluster.points:
                    if last_top_stixel_x is not None and abs(point[0] - last_top_stixel_x) > x_threshold:
                        top_stixel = Stixel(point=point, position_class=PositionClass.TOP)  # 1 for Top
                        cluster.stixels.append(top_stixel)
                        last_top_stixel_x = point[0]  # Update the x position for the next iteration
            # Add the top point of every cluster as a top stixel
            top_point = cluster.points[-1]  # Last point after sorting by z (highest z)
            top_stixel = Stixel(point=top_point, position_class=PositionClass.TOP)  # 1 for Top
            cluster.stixels.append(top_stixel)
            last_top_stixel_x = top_point[0]  # Update the x position for the next iteration

            # Update the previous cluster to the current one
            previous_cluster = cluster

    def get_stixels(self) -> List[Stixel]:
        stixels = [stixel for cluster in self.objects for stixel in cluster.stixels]
        return stixels

    def get_bottom_stixels(self) -> np.array:
        # Extract bottom stixels from all clusters using a list comprehension
        bottom_stixels = [stixel.point for cluster in self.objects for stixel in cluster.stixels if stixel.position_class == PositionClass.BOTTOM]
        return np.asarray(bottom_stixels)

    def get_top_stixels(self) -> np.array:
        # Extract bottom stixels from all clusters using a list comprehension
        top_stixels = [stixel.point for cluster in self.objects for stixel in cluster.stixels if stixel.position_class == PositionClass.TOP]
        return np.asarray(top_stixels)


def force_stixel_into_image_grid(stixels: List[List[Stixel]], image_size: Tuple[int,int], grid_step: int = 8) -> List[Stixel]:
    """
    Forces all given stixel into the output grid.
    Args:
        stixels:
        image_size:
        grid_step:
    Returns: a list of views with grid stixel
    """
    stixels = [item for sublist in stixels for item in sublist]
    # stacked_stixel = np.vstack(view_stixel)
    for stixel in stixels:
        stixel.point['proj_x'] = normalize_into_grid(stixel.point['proj_x'], step=grid_step)
        if stixel.point['proj_x'] == image_size[0]:
            stixel.point['proj_x'] = image_size[0] - grid_step
        stixel.point['proj_y'] = normalize_into_grid(stixel.point['proj_y'], step=grid_step)
        if stixel.point['proj_y'] == image_size[1]:
            stixel.point['proj_y'] = image_size[1] - grid_step
    return stixels


def normalize_into_grid(pos: int, step: int = 8, offset: int = 0):
    val_norm = 0
    rest = pos % step
    if rest > step / 2:
        val_norm = pos + (step - rest)
    else:
        val_norm = pos - rest
    assert val_norm % step == 0
    return val_norm + offset
