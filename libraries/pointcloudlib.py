import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
from libraries.names import point_dtype, StixelClass
import yaml
from scipy.spatial import distance
from libraries.helper import calculate_bottom_stixel_to_ground, calculate_bottom_stixel_by_line_of_sight


with open('libraries/pcl-config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def remove_ground(points: np.array) -> np.array:
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack((points['x'], points['y'], points['z'])).T
    pcd.points = o3d.utility.Vector3dVector(xyz)
    z_filter = (xyz[:, 2] <= config['rm_gnd']['z_max'])
    filtered_xyz = xyz[z_filter]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
    plane_model, inliers = filtered_pcd.segment_plane(
        distance_threshold=config['rm_gnd']['distance_threshold'],
        ransac_n=config['rm_gnd']['ransac_n'],
        num_iterations=config['rm_gnd']['num_iterations'])
    inlier_mask = np.zeros(len(xyz), dtype=bool)
    inlier_mask[np.where(z_filter)[0][inliers]] = True
    filtered_points = points[~inlier_mask]
    # determine sensor height
    ground_points = filtered_xyz[inliers]
    ground_pos = np.mean(ground_points[:, 2])
    return filtered_points, ground_pos


def remove_line_of_sight(points: np.array, camera_pose=None):
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
    def __init__(self, top_point: np.array, bottom_point: np.array, position_class: int, image_size: Dict[str, int], grid_step: int = 8):
        self.top_point: np.array = top_point
        self.position_class: int = position_class
        self.depth = self.calculate_depth()
        self.image_size = image_size
        self.grid_step = grid_step
        if position_class == StixelClass.OBJECT:
            self.bottom_point = top_point.copy()
            self.bottom_point['z'] = bottom_point['z']
            self.bottom_point['proj_y'] = bottom_point['proj_y']
        else:
            self.bottom_point: np.array = bottom_point
        self.force_stixel_to_grid()

    def force_stixel_to_grid(self):
        for point in (self.top_point, self.bottom_point):
            assert point['proj_x'] <= self.image_size['width'], f"x-value out of bounds ({point['proj_x']},{point['proj_y']})."
            point['proj_x'] = self._normalize_into_grid(point['proj_x'], step=self.grid_step)
            if point['proj_x'] == self.image_size['width']:
                point['proj_x'] = self.image_size['width'] - self.grid_step
            point['proj_y'] = self._normalize_into_grid(point['proj_y'], step=self.grid_step)
            if point['proj_y'] >= self.image_size['height']:
                point['proj_y'] = self.image_size['height'] - self.grid_step
            assert point['proj_y'] <= self.image_size['height'], f"y-value out of bounds ({point['proj_x']},{point['proj_y']})."

    @staticmethod
    def _normalize_into_grid(pos: int, step: int = 8):
        val_norm = pos - (pos % step)
        return val_norm

    def calculate_depth(self):
        depth = np.sqrt(self.top_point['x'] ** 2 + self.top_point['y'] ** 2 + self.top_point['z'] ** 2)
        return depth


class Cluster:
    def __init__(self, points: np.array, sensor_height: float):
        self.points: np.array = points  # Shape: x, y, z, proj_x, proj_y
        self.mean_range: float = self.calculate_mean_range()
        self.sensor_height = sensor_height
        self.stixels: List[Stixel] = []
        self.ground_reference_height, self.is_standing_on_ground = self.check_object_position()

    def __len__(self) -> int:
        return len(self.points)

    def calculate_mean_range(self) -> float:
        distances: List[float] = [np.sqrt(point['x'] ** 2 + point['y'] ** 2) for point in self.points]
        return float(np.mean(distances))

    def sort_points_bottom_stixel(self):
        # Sort points by ascending z
        self.points = sorted(self.points, key=lambda point: point['z'])

    def sort_points_top_obj_stixel(self):
        self.points = sorted(self.points, key=lambda point: point['z'], reverse=True)

    def check_object_position(self):
        self.sort_points_top_obj_stixel()
        cluster_point_with_lowest_z = self.points[-1]['z']
        if cluster_point_with_lowest_z - config['cluster']['to_ground_detection_threshold'] - self.sensor_height < 0:
            return self.sensor_height, True
        else:
            return cluster_point_with_lowest_z, False



def _euclidean_distance_with_raising_eps(p1, p2):
    # Berechnen Sie die normale euklidische Distanz zwischen den 2D-Punkten
    dist = distance.euclidean(p1, p2)
    # Der 'eps'-Wert basiert auf der 'range'-Komponente (erste Spalte) der Punkte
    dynamic_eps = config['scanline_cluster_obj']['clustering_factor'] * max(p1[0], p2[0]) + \
                  config['scanline_cluster_obj']['clustering_offset']
    # Überprüfen Sie, ob die Distanz innerhalb des dynamischen 'eps'-Wertes liegt
    return dist if dist <= dynamic_eps else np.inf


class Scanline:
    def __init__(self, points: np.array, camera_mtx, camera_position, camera_orientation, sensor_height, image_size):
        self.camera_mtx = camera_mtx
        self.camera_pov = camera_position
        self.camera_pose = camera_orientation
        self.sensor_height = sensor_height
        self.image_size = image_size
        self.points: np.array = np.array(points, dtype=point_dtype)
        self.objects: List[Cluster] = []
        self.last_object_top_stixel = None

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
            self.objects.append(Cluster(cluster_points, self.sensor_height))
        # Sort the list of Cluster objects by their mean_range
        self.objects = sorted(self.objects, key=lambda cluster: cluster.mean_range)

    def _determine_stixel(self):
        for cluster in self.objects:
            # cluster.sort_points_bottom_stixel()
            cluster.sort_points_top_obj_stixel()
            last_top_stixel_x: Stixel = None  # saves last Top-Stixel
            # add the top point
            for point in cluster.points:
                if last_top_stixel_x is None:
                    top_point = point
                    # sensor_height
                    if self.last_object_top_stixel is None:
                        bottom_point = calculate_bottom_stixel_to_ground(top_point, cluster.ground_reference_height,
                                                                         self.camera_pov, self.camera_pose,
                                                                         self.camera_mtx)
                    else:
                        bottom_point = calculate_bottom_stixel_by_line_of_sight(top_point, self.last_object_top_stixel.top_point,
                                                                                self.camera_pov, self.camera_pose,
                                                                                self.camera_mtx)
                    new_stixel = Stixel(top_point=top_point, bottom_point=bottom_point, position_class=StixelClass.TOP, image_size=self.image_size)
                    cluster.stixels.append(new_stixel)
                    last_top_stixel_x = new_stixel
                    if cluster.is_standing_on_ground:
                        self.last_object_top_stixel = new_stixel
                # Iterate through each point in the cluster to check the x_threshold condition
                elif (last_top_stixel_x.top_point['x'] - point['x']) > config['scanline_determine_stixel']['x_threshold']:
                    top_point = point
                    obj_stixel = Stixel(top_point=top_point, bottom_point=last_top_stixel_x.bottom_point, position_class=StixelClass.OBJECT, image_size=self.image_size)
                    cluster.stixels.append(obj_stixel)
                    last_top_stixel_x = obj_stixel   # Update the x position for the next iteration

    def get_stixels(self) -> List[Stixel]:
        self._cluster_objects()
        self._determine_stixel()
        stixels = [stixel for cluster in self.objects for stixel in cluster.stixels]
        return stixels


class StixelGenerator:
    def __init__(self, camera_mtx, camera_position, camera_orientation, img_size, sensor_height):
        self.camera_mtx = camera_mtx
        self.camera_pov = camera_position
        self.camera_pose = camera_orientation
        self.sensor_height = sensor_height
        self.img_size = img_size
        self.laser_scanlines = []

    def generate_stixel(self, laser_points: np.array) -> List[Stixel]:
        laser_points_by_angle = group_points_by_angle(laser_points)
        stixels = []
        for angle_laser_points in laser_points_by_angle:
            column = Scanline(angle_laser_points,
                              camera_mtx=self.camera_mtx,
                              camera_position=self.camera_pov,
                              camera_orientation=self.camera_pose,
                              sensor_height=self.sensor_height,
                              image_size=self.img_size)
            stixels.append(column.get_stixels())
        return [stixel for sublist in stixels for stixel in sublist]
