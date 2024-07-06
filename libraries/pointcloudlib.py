import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
from libraries.names import point_dtype, point_dtype_ext, StixelClass
import yaml
from scipy.spatial import distance
from libraries.helper import BottomPointCalculator
from dataloader.Stixel import BaseStixel


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
    return filtered_points, plane_model


def remove_line_of_sight(points: np.array, camera_pose=None):
    # Manually extract x, y, z raw from the structured array
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

def remove_pts_below_plane_model(points: np.array, plane_model) -> np.array:
    a, b, c, d = plane_model
    filtered_points = []
    for point in points:
        x, y, z, proj_x, proj_y = point
        if a * x + b * y + c * z + d >= 0:
            filtered_points.append(point)
    return np.array(filtered_points)


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
    def __init__(self, top_point: np.array, bottom_point: np.array, position_class: StixelClass, image_size: Dict[str, int], grid_step: int = 8):
        self.column = top_point['proj_x']
        self.top_row = top_point['proj_y']
        self.bottom_row = bottom_point['proj_y']
        self.position_class: StixelClass = position_class
        self.top_point = top_point
        self.bottom_point = bottom_point
        self.depth = self.calculate_depth(top_point)
        self.image_size = image_size
        self.grid_step = grid_step

        self.force_stixel_to_grid()
        self.check_integrity()

    def force_stixel_to_grid(self):
        for attr in ('top_row', 'bottom_row'):
            normalized_row = self._normalize_into_grid(getattr(self, attr), step=self.grid_step)
            if normalized_row >= self.image_size['height']:
                normalized_row = self.image_size['height'] - self.grid_step
            setattr(self, attr, normalized_row)
        if self.top_row == self.bottom_row:
            if self.top_row == self.image_size['height'] - self.grid_step:
                self.top_row -= self.grid_step
            else:
                self.bottom_row += self.grid_step
        self.column = self._normalize_into_grid(self.column, step=self.grid_step)
        if self.column == self.image_size['width']:
            self.column = self.image_size['width'] - self.grid_step

    def check_integrity(self):
        for cut_row in (self.top_row, self.bottom_row):
            assert cut_row <= self.image_size['height'], f"y-value out of bounds ({self.column},{cut_row})."
            assert cut_row % self.grid_step == 0, f"y-value is not into grid ({self.column},{cut_row})."
        assert self.column <= self.image_size['width'], f"x-value out of bounds ({self.column},{cut_row})."
        assert self.column % self.grid_step == 0, f"x-value is not into grid ({self.column},{cut_row})."
        #assert self.top_row < self.bottom_row, f"Top is higher than Bottom. Top_pt: {self.top_point}. Bottom_pt:{self.bottom_point}."
        assert self.top_row != self.bottom_row, "Top is Bottom."

    @staticmethod
    def _normalize_into_grid(pos: int, step: int = 8):
        val_norm = pos - (pos % step)
        return val_norm

    @staticmethod
    def calculate_depth(top_point):
        depth = np.sqrt(top_point['x'] ** 2 + top_point['y'] ** 2 + top_point['z'] ** 2)
        return depth


class Cluster:
    def __init__(self, points: np.array, plane_model):
        self.plane_model = plane_model
        self.points: np.array = points
        self.is_standing_on_ground = self.check_object_position()
        if self.is_standing_on_ground:
            self.points: np.array = self.assign_reference_z_to_points_from_ground(points)    # Shape: x, y, z, proj_x, proj_y, z_ref
        else:
            self.points: np.array = self.assign_reference_z_to_points_from_object_low(points)
        self.mean_range: float = self.calculate_mean_range()
        self.stixels: List[Stixel] = []


    def __len__(self) -> int:
        return len(self.points)

    def calculate_mean_range(self) -> float:
        distances: List[float] = [np.sqrt(point['x'] ** 2 + point['y'] ** 2) for point in self.points]
        return float(np.mean(distances))

    def sort_points_bottom_stixel(self):
        # Sort points by ascending z: -3.2, -2.0, 0.65, ...
        self.points = sorted(self.points, key=lambda point: point['z'])

    def sort_points_top_obj_stixel(self):
        # Sort points by descending z: 3.2, 2.0, 0.65, ...
        self.points = sorted(self.points, key=lambda point: point['z'], reverse=True)

    def assign_reference_z_to_points_from_ground(self, points):
        referenced_points = np.empty(points.shape, dtype=point_dtype_ext)
        a, b, c, d = self.plane_model
        #d -= 0.05
        assert c != 0, "Dont divide by 0"
        for i, point in enumerate(points):
            x, y, z, proj_x, proj_y = point
            z_ref = -(a * x + b * y + d) / c
            referenced_points[i] = (x, y, z, proj_x, proj_y, z_ref)
        return referenced_points

    def assign_reference_z_to_points_from_object_low(self, points):
        referenced_points = np.empty(points.shape, dtype=point_dtype_ext)
        self.sort_points_top_obj_stixel()
        cluster_point_ref_z = self.points[-1]['z']
        for i, point in enumerate(points):
            x, y, z, proj_x, proj_y = point
            referenced_points[i] = (x, y, z, proj_x, proj_y, cluster_point_ref_z)
        return referenced_points

    def check_object_position(self):
        self.sort_points_top_obj_stixel()
        cluster_point = self.points[-1]
        a, b, c, d = self.plane_model
        # calculate distance to plane
        distance = abs(a * cluster_point['x'] + b * cluster_point['y'] + c * cluster_point['z'] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        if distance <= config['cluster']['to_ground_detection_threshold']:
            return True
        else:
            return False



def _euclidean_distance_with_raising_eps(p1, p2):
    # Berechnen Sie die normale euklidische Distanz zwischen den 2D-Punkten
    dist = distance.euclidean(p1, p2)
    # Der 'eps'-Wert basiert auf der 'range'-Komponente (erste Spalte) der Punkte
    dynamic_eps = config['scanline_cluster_obj']['clustering_factor'] * max(p1[0], p2[0]) + \
                  config['scanline_cluster_obj']['clustering_offset']
    # Überprüfen Sie, ob die Distanz innerhalb des dynamischen 'eps'-Wertes liegt
    return dist if dist <= dynamic_eps else np.inf


class Scanline:
    def __init__(self, points: np.array, camera_info, plane_model, image_size):
        self.camera_info = camera_info
        self.plane_model = plane_model
        self.bottom_pt_calc = BottomPointCalculator(camera_xyz=self.camera_info.extrinsic.xyz,
                                                    camera_rpy=self.camera_info.extrinsic.rpy,
                                                    camera_mtx=self.camera_info.camera_mtx,
                                                    proj_mtx=self.camera_info.projection_mtx,
                                                    rect_mtx=self.camera_info.rectification_mtx)
        self.image_size = image_size
        self.points: np.array = np.array(points, dtype=point_dtype)
        self.objects: List[Cluster] = []
        self.last_cluster_top_stixel = None

    def _cluster_objects(self):
        # Compute the radial distance r
        r_values = np.sqrt(self.points['x'] ** 2 + self.points['y'] ** 2)
        # Sort points by r for clustering
        sorted_indices = np.argsort(r_values)
        sorted_r = r_values[sorted_indices]
        sorted_z = self.points['z'][sorted_indices]
        self.points = self.points[sorted_indices]
        # Prepare the raw for DBSCAN
        db_data = np.column_stack((sorted_r, sorted_z))
        # Check if enough raw points are present for clustering
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
            self.objects.append(Cluster(cluster_points, self.plane_model))
        # Sort the list of Cluster objects by their mean_range
        self.objects = sorted(self.objects, key=lambda cluster: cluster.mean_range)

    def _determine_stixel(self):
        for cluster in self.objects:
            # cluster.sort_points_bottom_stixel()
            cluster.sort_points_top_obj_stixel()
            last_cluster_stixel_x: Stixel = None  # saves last Top-Stixel
            # add the top point
            for point in cluster.points:
                top_point = None
                bottom_point = None
                point_dist = np.sqrt(point['x'] ** 2 + point['y'] ** 2)

                if last_cluster_stixel_x is None:
                    last_stixel_dist = None
                else:
                    last_stixel_dist = np.sqrt(
                        last_cluster_stixel_x.top_point['x'] ** 2 + last_cluster_stixel_x.top_point['y'] ** 2)

                if (last_cluster_stixel_x is None or
                        (last_stixel_dist is not None and (last_stixel_dist - point_dist) >
                         config['scanline_determine_stixel']['x_threshold'])):
                    top_point = point
                    # sensor_height
                    if self.last_cluster_top_stixel is None:
                        bottom_point = self.bottom_pt_calc.calculate_bottom_stixel_to_reference_height(top_point)
                    else:
                        bottom_point = self.bottom_pt_calc.calculate_bottom_stixel_by_line_of_sight(top_point,
                                                                                                    self.last_cluster_top_stixel.top_point)
                    pos_cls = StixelClass.TOP if last_cluster_stixel_x is None else StixelClass.OBJECT
                    new_stixel = Stixel(top_point=top_point, bottom_point=bottom_point, position_class=pos_cls, image_size=self.image_size)
                    cluster.stixels.append(new_stixel)
                    last_cluster_stixel_x = new_stixel
                    if cluster.is_standing_on_ground and last_cluster_stixel_x is None:
                        self.last_cluster_top_stixel = new_stixel

    def get_stixels(self) -> List[Stixel]:
        self._cluster_objects()
        self._determine_stixel()
        stixels = [stixel for cluster in self.objects for stixel in cluster.stixels]
        return stixels


class StixelGenerator:
    def __init__(self, camera_info, img_size, plane_model):
        # self.camera_mtx = camera_mtx
        # self.camera_pov = camera_position
        # self.camera_pose = camera_orientation
        self.camera_info = camera_info
        self.plane_model = plane_model
        self.img_size = img_size
        self.laser_scanlines = []

    def generate_stixel(self, laser_points: np.array) -> List[Stixel]:
        laser_points_by_angle = group_points_by_angle(laser_points)
        stixels = []
        for angle_laser_points in laser_points_by_angle:
            column = Scanline(angle_laser_points,
                              camera_info=self.camera_info,
                              plane_model=self.plane_model,
                              image_size=self.img_size)
            stixels.append(column.get_stixels())
        return [stixel for sublist in stixels for stixel in sublist]
