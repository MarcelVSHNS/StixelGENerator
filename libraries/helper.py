from typing import List, Dict, Tuple
import numpy as np


def _euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix."""
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_x = np.array([[1, 0, 0],
                    [0, cos_r, -sin_r],
                    [0, sin_r, cos_r]])
    R_y = np.array([[cos_p, 0, sin_p],
                    [0, 1, 0],
                    [-sin_p, 0, cos_p]])
    R_z = np.array([[cos_y, -sin_y, 0],
                    [sin_y, cos_y, 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def _create_transformation_matrix(translation, rotation):
    """Create a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = _euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
    T[:3, 3] = translation
    return T


def _transform_to_sensor(camera_extrinsic: Tuple[np.array, np.array]):
    """Transform the data to the sensor's coordinate frame."""
    lidar_pov = np.array([0, 0, 0])
    lidar_pose = np.array([0, 0, 0])
    t1 = _create_transformation_matrix(lidar_pov, lidar_pose)
    t2 = _create_transformation_matrix(camera_extrinsic[0], camera_extrinsic[1])
    t2_to_1 = np.dot(np.linalg.inv(t2), t1)
    return t2_to_1


def project_point_into_image(point: np.ndarray, camera_pov: np.array, camera_pose: np.array, camera_mtx: np.array) -> Tuple[int, int]:
    """Retrieve the projection matrix based on provided parameters."""
    point = np.stack([point['x'], point['y'], point['z']], axis=-1)
    lidar_to_cam_tf_mtx = _transform_to_sensor((camera_pov, camera_pose))
    point_in_camera = np.dot(lidar_to_cam_tf_mtx, np.append(point[:3], 1))  # Nehmen Sie nur die ersten 3 Koordinaten
    pixel = np.dot(camera_mtx, point_in_camera[:3])
    u, v = int(pixel[0] / pixel[2]), int(pixel[1] / pixel[2])
    v += 35
    projection = (u, v)
    return projection


def calculate_bottom_stixel_to_ground(top_point: np.array, sensor_height: float, camera_pov: np.array, camera_pose: np.array, camera_mtx: np.array, apply_gnd_offset=False) -> np.array:
    bottom_point = top_point.copy()
    bottom_point['z'] = sensor_height
    x_proj, y_proj = project_point_into_image(bottom_point, camera_pov=camera_pov, camera_pose=camera_pose,
                                          camera_mtx=camera_mtx)
    #assert x_proj == top_point['proj_x']
    if apply_gnd_offset:
        m = -0.25
        b = 15
        range = np.sqrt(bottom_point['x'] ** 2 + bottom_point['y'] ** 2)
        offset = int(m * range + b)
        if offset < 0:
            offset = 0
        if y_proj + offset < 1200:
            y_proj += offset
        else:
            y_proj = 1200
    bottom_point['proj_y'] = y_proj
    return bottom_point


def find_linear_equation(pt1: Tuple[float, float], pt2: Tuple[float, float]):
    x1, y1 = pt1
    x2, y2 = pt2
    m = (y2 - y1) / (x2 - x1)  # Steigung berechnen
    b = y1 - m * x1  # y-Achsenabschnitt berechnen
    return m, b


def get_range(x: float,y: float) -> float:
    return np.sqrt(x ** 2 + y ** 2)


def calculate_bottom_stixel_by_line_of_sight(top_point: np.array, last_point: np.array, camera_pov: np.array, camera_pose: np.array, camera_mtx: np.array, los_offset=0) -> np.array:
    bottom_point = top_point.copy()
    camera_pt = (get_range(camera_pov[0], camera_pov[1]), camera_pov[2])
    last_stixel_pt = (get_range(last_point['x'], last_point['y']), last_point['z'])
    m, b = find_linear_equation(camera_pt, last_stixel_pt)
    bottom_point_range = get_range(bottom_point['x'], bottom_point['y'])
    new_bottom_z = (m * bottom_point_range + b)
    bottom_point['z'] = new_bottom_z
    x_proj, y_proj = project_point_into_image(bottom_point, camera_pov=camera_pov, camera_pose=camera_pose,
                                              camera_mtx=camera_mtx)
    # assert x_proj == top_point['proj_x']
    bottom_point['proj_y'] = y_proj - los_offset
    return bottom_point
