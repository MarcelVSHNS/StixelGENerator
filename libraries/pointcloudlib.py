import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from libraries.names import point_dtype, PositionClass
import yaml
from scipy.spatial import distance


with open('libraries/pcl-config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def remove_ground(points: np.array) -> np.array:
    # Konvertieren Sie das Eingabearray in eine Punktwolke
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack((points['x'], points['y'], points['z'])).T
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Filtern der Punkte basierend auf dem Z-Bereich
    z_filter = (xyz[:, 2] <= config['rm_gnd']['z_max'])
    filtered_xyz = xyz[z_filter]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_xyz)

    # RANSAC auf die gefilterte Punktwolke anwenden
    plane_model, inliers = filtered_pcd.segment_plane(
        distance_threshold=config['rm_gnd']['distance_threshold'],
        ransac_n=config['rm_gnd']['ransac_n'],
        num_iterations=config['rm_gnd']['num_iterations'])

    # Erstellen Sie eine boolesche Maske für Inliers in der gefilterten Punktwolke
    inlier_mask = np.zeros(len(xyz), dtype=bool)
    inlier_mask[np.where(z_filter)[0][inliers]] = True  # Setzen Sie Inliers in der ursprünglichen Punktwolke

    # Filtern Sie die Punkte, indem Sie die Bodenpunkte (Inliers) überspringen
    filtered_points = points[~inlier_mask]
    return filtered_points


def remove_line_of_sight(points: np.array, camera_pose=None) :
    # Manually extract x, y, z data from the structured array
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack((points['x'], points['y'], points['z'])).T
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if camera_pose is None:
        camera_pose = [config['rm_los']['x'],
                  config['rm_los']['y'],
                  config['rm_los']['z']]
    radius = config['rm_los']['radius']
    _, pt_map = pcd.hidden_point_removal(camera_pose, radius)
    mask = np.zeros(len(np.asarray(points)), dtype=bool)
    mask[pt_map] = True
    filtered_points = points[mask]
    return filtered_points


def remove_far_points(points: np.array) -> np.array:
    # Calculate the distance (range) from x and y for each point
    ranges = np.sqrt(points['x'] ** 2 + points['y'] ** 2)
    # Filter out points where the distance is greater than the threshold
    filtered_points = points[ranges <= config['rm_far_pts']['range_threshold']]
    return filtered_points


def group_points_by_angle(points: np.array) -> List[np.array]:
    """
    Groups points based on their azimuth angle and returns a list of arrays,
    each containing the points (x, y, z, proj_x, proj_y) of the same angle.

    :param points: A numpy array of points (x, y, z, proj_x, proj_y).
    :param eps: The maximum distance between two points to be considered in the same cluster.
    :param min_samples: The number of points required in a neighborhood for a point to be considered a core point.
    :return: A list of numpy arrays, each containing points of an angle cluster.
    """
    # Compute the azimuth angle for each point
    azimuth_angles = np.arctan2(points['y'], points['x'])
    sorted_pairs = sorted(zip(points, azimuth_angles), key=lambda x: x[1])
    sorted_points = np.array([point for point, angle in sorted_pairs])
    sorted_azimuth_angles = np.array([angle for point, angle in sorted_pairs])
    # azimuth_angles = np.mod(azimuth_angles, 2 * np.pi)
    # Perform DBSCAN clustering based on azimuth angles only
    db = DBSCAN(eps=config['group_angle']['eps'],
                min_samples=config['group_angle']['min_samples']).fit(sorted_azimuth_angles.reshape(-1, 1))
    # Obtain cluster labels
    labels = db.labels_
    # Group points based on labels and compute average angles
    # create a list for every found cluster, if you found not matched points subtract 1
    angle_cluster = [[] for _ in range(len(set(labels)) - (1 if -1 in labels else 0))]
    # fill in the points
    for point, label in zip(sorted_points, labels):
        if label != -1:  # if you have not clustered points
            angle_cluster[label].append(point)
    return angle_cluster


class Stixel:
    def __init__(self, point: np.array, position_class: PositionClass):
        self.point: np.array = point
        self.point['proj_y'] += config['stixel']['proj_y_offset']
        self.position_class: PositionClass = position_class
        # If stixel is a bottom stixel...
        if self.position_class == PositionClass.BOTTOM:
            # ... apply the floor offset (in px) to proj_y
            range = np.sqrt(self.point['x']**2 + self.point['y']**2)
            offset = int(config['stixel']['ground_offset_m'] * range + config['stixel']['ground_offset_b'])
            self.point['proj_y'] += (offset if offset > 0 else 0)


def normalize_into_grid(pos: int, step: int = 8):
    """val_norm = 0
    rest = pos % step
    if rest > step / 2:
        val_norm = pos + (step - rest)
    else:
        val_norm = pos - rest
    assert val_norm % step == 0
    return val_norm"""
    val_norm = pos - (pos % step)
    return val_norm


def force_stixel_into_image_grid(stixels: List[List[Stixel]], image_size: Dict[str, int], grid_step: int = 8) -> List[Stixel]:
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
        if stixel.point['proj_x'] == image_size['width']:
            stixel.point['proj_x'] = image_size['width'] - grid_step
        stixel.point['proj_y'] = normalize_into_grid(stixel.point['proj_y'], step=grid_step)
        if stixel.point['proj_y'] == image_size['height']:
            stixel.point['proj_y'] = image_size['height'] - grid_step
    return stixels


class Cluster:
    def __init__(self, points: np.array):
        self.points: np.array = points            # Shape: x, y, z, proj_x, proj_y
        self.mean_range: float = self.calculate_mean_range()
        self.stixels: List[Stixel] = []

    def __len__(self) -> int:
        return len(self.points)

    def calculate_mean_range(self) -> float:
        distances: List[float] = [np.sqrt(point['x']**2 + point['y']**2) for point in self.points]
        return np.mean(distances)

    def sort_points_bottom_stixel(self):
        # Sort points by ascending z
        self.points = sorted(self.points, key=lambda point: point['z'])

    def sort_points_top_obj_stixel(self):
        # Sort points by ascending z
        self.points = sorted(self.points, key=lambda point: point['z'], reverse=True)


def _euclidean_distance_with_raising_eps(p1, p2):
    # TODO: implement custom function for dbcsan at object clustering
    # Berechnen Sie die normale euklidische Distanz zwischen den 2D-Punkten
    dist = distance.euclidean(p1, p2)
    # Der 'eps'-Wert basiert auf der 'range'-Komponente (erste Spalte) der Punkte
    dynamic_eps = config['scanline_cluster_obj']['clustering_factor'] * max(p1[0], p2[0]) + config['scanline_cluster_obj']['clustering_offset']
    # Überprüfen Sie, ob die Distanz innerhalb des dynamischen 'eps'-Wertes liegt
    return dist if dist <= dynamic_eps else np.inf


class Scanline:
    def __init__(self, points: np.array):
        self.points: np.array = np.array(points, dtype=point_dtype)
        self.objects: List[Cluster] = []
        self._cluster_objects()
        self._determine_stixel()

    def _cluster_objects(self):
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
            db = DBSCAN(eps=100,
                        min_samples=config['scanline_cluster_obj']['min_samples'],
                        metric=_euclidean_distance_with_raising_eps).fit(db_data)
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

    def _determine_stixel(self):
        for cluster in self.objects:
            num_pts = len(cluster)
            cluster.sort_points_bottom_stixel()
            # Add the bottom point of the first cluster or if the distance to the previous cluster is larger than the threshold
            bottom_point = cluster.points[0]  # First point after sorting by z (lowest z)
            bottom_stixel = Stixel(point=bottom_point, position_class=PositionClass.BOTTOM)
            cluster.stixels.append(bottom_stixel)
            #cluster.points.pop(0)     # remove bottom point from list

            cluster.sort_points_top_obj_stixel()
            last_top_stixel_x = None  # saves last Top-Stixel
            # add the top point
            for point in cluster.points[:-1]:
                if last_top_stixel_x is None:
                    top_point = cluster.points[0]
                    cluster.stixels.append(Stixel(point=top_point, position_class=PositionClass.TOP))
                    last_top_stixel_x = point['x']
                # Iterate through each point in the cluster to check the x_threshold condition
                elif (last_top_stixel_x - point['x']) > config['scanline_determine_stixel']['x_threshold']:
                    obj_stixel = Stixel(point=point, position_class=PositionClass.OBJECT)
                    cluster.stixels.append(obj_stixel)
                    last_top_stixel_x = point['x']  # Update the x position for the next iteration

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
