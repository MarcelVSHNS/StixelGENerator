import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List
from libraries.Stixel import point_dtype, point_dtype_ext, StixelClass
import yaml
from scipy.spatial import distance
from libraries.helper import BottomPointCalculator
from libraries import Stixel

with open('libraries/pcl-config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def remove_ground(points: np.array) -> np.array:
    """
    Args:
        points: A numpy array of shape (N, 3) representing the 3D coordinates of points.
    Returns:
        filtered_points: A numpy array of shape (M, 3) containing the filtered points after removing the ground.
        plane_model: A tuple containing the parameters of the detected ground plane.
    """
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
    """
    Removes points that are in line of sight of a camera.
    Args:
        points: A numpy array containing structured data with fields 'x', 'y', 'z'.
        camera_pose: Optional camera pose as a list of [x, y, z]. If not provided, it will use the default camera pose
        from the configuration.
    Returns:
        filtered_points: A numpy array containing structured data with the same fields as 'points', with the points
        that are in line of sight removed.
    """
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
    """
    Args:
        points (np.array): An array of points, with each point represented as a structured array containing the 'x' and
         'y' coordinates.
    Returns:
        np.array: An array of points filtered to remove points that are far away from the origin.
    Description:
    This method takes in an array of points and calculates the distance from the origin (0,0) for each point using the
    Euclidean distance formula. It then filters out points where the distance is greater than a specified threshold,
    which is retrieved from the 'config' dictionary using the key 'rm_far_pts.range_threshold'.
    Example:
        points = np.array([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
        filtered_points = remove_far_points(points)
        # Output: array([(1, 2), (3, 4), (5, 6)])
    Note:
        - The input `points` array should be a structured array with fields 'x' and 'y'.
        - The 'config' dictionary should contain the 'rm_far_pts.range_threshold' key, which specifies the maximum
          distance allowed between a point and the origin.
        - The return value is an array of points where the distance from the origin is less than or equal to the
          threshold.
    """
    # Calculate the distance (range) from x and y for each point
    ranges = np.sqrt(points['x'] ** 2 + points['y'] ** 2)
    # Filter out points where the distance is greater than the threshold
    filtered_points = points[ranges <= config['rm_far_pts']['range_threshold']]
    return filtered_points


def remove_pts_below_plane_model(points: np.array, plane_model) -> np.array:
    """
    Removes points that are below the plane model from the given array of points.
    Args:
        points (numpy.array): An array of points, where each point is represented by an array of five values
        (x, y, z, proj_x, proj_y).
        plane_model: A tuple representing the coefficients of the plane model (a, b, c, d), where a, b, c are the
        normal vector components of the plane and d is the distance from the origin.
    Returns:
        numpy.array: An array of points that are above or on the plane model"""
    a, b, c, d = plane_model
    filtered_points = []
    for point in points:
        x, y, z, proj_x, proj_y = point
        if a * x + b * y + c * z + d >= 0:
            filtered_points.append(point)
    return np.array(filtered_points)


def group_points_by_angle(points: np.array) -> List[np.array]:
    """
    Groups points based on their azimuth angles.
    Args:
        points: A numpy array of points, where each point has 'x' and 'y' coordinates.
    Returns:
        A list of numpy arrays, where each array represents a cluster of points with similar azimuth angles.
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


class Cluster:
    """
    Class representing a cluster of points.
    Attributes:
    - points (np.array): Array of points in the cluster. Shape: x, y, z, proj_x, proj_y, z_ref
    - plane_model (tuple): Tuple of four coefficients (a, b, c, d) representing the plane model of the cluster
    Methods:
    - __len__() -> int: Returns the number of points in the cluster
    - calculate_mean_range() -> float: Calculates the mean range of the points in the cluster
    - sort_points_bottom_stixel(): Sorts the points by ascending z value
    - sort_points_top_obj_stixel(): Sorts the points by descending z value
    - assign_reference_z_to_points_from_ground(points): Assigns reference z values to points from the ground
    - assign_reference_z_to_points_from_object_low(points): Assigns reference z values to points from the lowest object
     point
    - check_object_position() -> bool: Checks if the cluster is standing on the ground
    """
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
    """
    Calculates the Euclidean distance between two 2D points, with the possibility of raising the epsilon value.
    Args:
        p1: A tuple representing the coordinates of the first point.
        p2: A tuple representing the coordinates of the second point.

    Returns:
        The Euclidean distance between the two points if it is within the dynamic epsilon value, otherwise returns
         infinity.
    """
    dist = distance.euclidean(p1, p2)
    dynamic_eps = config['scanline_cluster_obj']['clustering_factor'] * max(p1[0], p2[0]) + \
                  config['scanline_cluster_obj']['clustering_offset']
    return dist if dist <= dynamic_eps else np.inf


class Scanline:
    """
    Class representing a scanline.
    Attributes:
        camera_info (object): Information about the camera used to capture the scanline.
        plane_model (object): Model representing the plane where objects are detected.
        bottom_pt_calc (object): Calculator for determining the bottom point of stixels.
        image_size (tuple): Size of the image.
    Methods:
        __init__(self, points: np.array, camera_info, plane_model, image_size):
            Initializes a new Scanline object.
        _cluster_objects(self):
            Clusters the objects in the scanline based on their radial distance from the camera.
        _determine_stixel(self):
            Determines the stixels for each clustered object in the scanline.
        get_stixels(self) -> List[Stixel]:
            Returns a list of stixels found in the scanline.
    """
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
    """
    StixelGenerator -> ScanLine -> Cluster -> Stixel
    Class representing a Stixel Generator.
    StixelGenerator is responsible for generating stixels from laser points.
    Attributes:
        camera_info (CameraInfo): The camera info object containing camera parameters.
        img_size (tuple): The size of the image (width, height).
        plane_model (PlaneModel): The plane model used for plane fitting.
        laser_scanlines (list): The list of laser scanlines.
    Methods:
        generate_stixel: Generates stixels from laser points.
    """
    def __init__(self, camera_info, img_size, plane_model):
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
