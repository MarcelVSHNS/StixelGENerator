from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataloader import CameraInfo
from libraries.Stixel import point_dtype, point_dtype_sph


class Transformation:
    """
    This class represents a transformation from one coordinate frame to another. It contains methods to set and
    retrieve information about the transformation.
    Attributes:
        _at (str): The name of the coordinate frame that the transformation is applied to.
        _to (str): The name of the coordinate frame that the transformation is applied from.
        _translation (np.ndarray): The translation vector of the transformation.
        _rotation (np.ndarray): The rotation vector of the transformation.
        transformation_mtx (np.ndarray): The transformation matrix representing the transformation.
    Methods:
        __init__(self, at, to, xyz, rpy)
            Initializes a new instance of the Transformation class.
        at (property)
            Getter and setter for the _at attribute.
        to (property)
            Getter and setter for the _to attribute.
        translation (property)
            Getter and setter for the _translation attribute.
        rotation (property)
            Getter and setter for the _rotation attribute.
        _update_transformation_matrix(self)
            Updates the transformation matrix based on the current translation and rotation vectors.
        add_transformation(self, transformation_to_add)
            Adds another transformation to the current transformation and returns a new Transformation object.
        invert_transformation(self)
            Inverts the current transformation and returns a new Transformation object representing the inverse transformation.
    """
    def __init__(self, at, to, xyz, rpy):
        self._at = at
        self._to = to
        self._translation = xyz
        self._rotation = rpy
        self._update_transformation_matrix()

    @property
    def at(self):
        return self._at

    @at.setter
    def at(self, value):
        self._at = value

    @property
    def to(self):
        return self._to

    @to.setter
    def to(self, value):
        self._to = value

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    def _update_transformation_matrix(self):
        """ Method to update the transformation matrix based on the current rotation and translation values. """
        rotation = R.from_euler('xyz', self._rotation, degrees=False)
        rotation_matrix = rotation.as_matrix()
        self.transformation_mtx = np.identity(4)
        self.transformation_mtx[:3, :3] = rotation_matrix
        self.transformation_mtx[:3, 3] = self._translation

    def add_transformation(self, transformation_to_add):
        transformation_mtx_to_add = transformation_to_add.transformation_mtx
        new_transformation_mtx = np.dot(self.transformation_mtx, transformation_mtx_to_add)
        translation_vector, euler_angles = extract_translation_and_euler_from_matrix(new_transformation_mtx)
        new_transformation = Transformation(self.at, transformation_to_add.to, translation_vector, euler_angles)
        return new_transformation

    def invert_transformation(self):
        inverse_rotation_matrix = self.transformation_mtx[:3, :3].T
        inverse_translation_vector = -inverse_rotation_matrix @ self.transformation_mtx[:3, 3]
        inverse_transformation_matrix = np.identity(4)
        inverse_transformation_matrix[:3, :3] = inverse_rotation_matrix
        inverse_transformation_matrix[:3, 3] = inverse_translation_vector
        translation_vector, euler_angles = extract_translation_and_euler_from_matrix(inverse_transformation_matrix)
        inverse_transformation = Transformation(self.to, self.at, translation_vector, euler_angles)
        return inverse_transformation


def extract_translation_and_euler_from_matrix(mtx):
    translation_vector = mtx[:3, 3]
    rotation_matrix = mtx[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles_rad = rotation.as_euler('xyz', degrees=False)
    return translation_vector, euler_angles_rad


def find_linear_equation(pt1: Tuple[float, float], pt2: Tuple[float, float]):
    x1, y1 = pt1
    x2, y2 = pt2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def get_range(x: float, y: float) -> float:
    return np.sqrt(x ** 2 + y ** 2)


def cart_2_sph(point: np.array):
    x, y, z = point['x'], point['y'], point['z']
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    point_sph = np.array(
        (r, az, el, point['proj_x'], point['proj_y'], point['sem_seg']),
        dtype=point_dtype_sph
    )
    return point_sph


def sph_2_cart(point: np.array):
    r, az, el = point['r'], point['az'], point['el']
    r_cos_theta = r * np.cos(el)
    x = r_cos_theta * np.cos(az)
    y = r_cos_theta * np.sin(az)
    z = r * np.sin(el)
    point_cart = np.array(
        (x, y, z, point['proj_x'], point['proj_y'], point['sem_seg']),
        dtype=point_dtype
    )
    return point_cart


class BottomPointCalculator:
    def __init__(self,
                 cam_info: CameraInfo,
                 los_offset=0,
                 apply_gnd_offset=False):
        self.camera_info = cam_info
        self.los_offset = los_offset
        self.apply_gnd_offset = apply_gnd_offset

    def calculate_bottom_stixel_by_line_of_sight(self, top_point: np.array, last_point: np.array) -> np.array:
        bottom_point = top_point.copy()
        camera_pt = (get_range(self.camera_info.extrinsic.xyz[0],
                               self.camera_info.extrinsic.xyz[1]),
                     self.camera_info.extrinsic.xyz[2])
        last_stixel_pt = (get_range(last_point['x'], last_point['y']), last_point['z'])
        m, b = find_linear_equation(camera_pt, last_stixel_pt)
        bottom_point_range = get_range(bottom_point['x'], bottom_point['y'])
        new_bottom_z = (m * bottom_point_range + b)
        bottom_point['z'] = new_bottom_z
        x_proj, y_proj = self.project_point_into_image(bottom_point)
        # assert x_proj == top_point['proj_x']
        bottom_point['proj_y'] = y_proj - self.los_offset
        return bottom_point

    def calculate_bottom_stixel_to_reference_height(self, top_point: np.array) -> np.array:
        bottom_point = top_point.copy()
        bottom_point['z'] = top_point['z_ref']
        x_proj, y_proj = self.project_point_into_image(bottom_point)
        # assert x_proj == top_point['proj_x']
        if self.apply_gnd_offset:
            m = -0.25
            b = 15
            range = np.sqrt(bottom_point['x'] ** 2 + bottom_point['y'] ** 2)
            offset = int(m * range + b)
            print(f'applied gnd offset: {offset}')
            if offset < 0:
                offset = 0
            if y_proj - offset > 0:
                y_proj -= offset
            else:
                y_proj = 0
        bottom_point['proj_y'] = y_proj
        return bottom_point

    def project_point_into_image(self, point: np.ndarray) -> Tuple[int, int]:
        """Retrieve the projection matrix based on provided parameters."""
        pt = np.stack([point['x'], point['y'], point['z']], axis=-1)
        # P = K * Rect * R|t with R|t = T
        # https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        point_in_camera = self.camera_info.P.dot(self.camera_info.R.dot(self.camera_info.T.dot(np.append(pt[:3], 1))))
        # pixel = np.dot(self.camera_mtx, point_in_camera[:3])
        u = int(point_in_camera[0] / point_in_camera[2])
        v = int(point_in_camera[1] / point_in_camera[2])
        projection = (u, v)
        return projection
