import numpy as np
import pykitti
import os
from typing import List, Tuple
from libraries.Stixel import point_dtype
from dataloader import BaseData, CameraInfo, Pose


class KittiData(BaseData):
    """
    Class representing KITTI dataset.
    This class contains information about a specific data sample in the KITTI dataset, including the RGB images,
    3D point cloud, and calibration data.
    Attributes:
        name (str): The name of the data sample.
        image (numpy.array): The left rectified RGB image.
        image_right (numpy.array): The right rectified RGB image.
        points (numpy.array): The 3D point cloud.
        t_cam2_velo (numpy.array): The transformation matrix from camera 2 coordinates to velodyne coordinates.
        camera_info (CameraInfo): The camera information object containing metadata and calibration data.
    Methods:
        __init__(name, rgb_stereo_pair, velo_scan, calib_data): Initializes a new KittiData instance with the given
        parameters.
        point_slices() -> numpy.array: Processes the point cloud and returns the filtered points with their respective
        pixel coordinates.
    """
    def __init__(self, name, rgb_stereo_pair, velo_scan, calib_data):
        super().__init__()
        self.name: str = name
        self.image = rgb_stereo_pair[0]     # cam2, rectified
        self.image_right = rgb_stereo_pair[1]       # cam3, rectified
        self.points: np.array = velo_scan
        self.c = calib_data
        self.camera_info: CameraInfo = CameraInfo(camera_mtx=calib_data.K_cam2,
                                                  proj_mtx=np.array(calib_data.P_rect_20),
                                                  rect_mtx=calib_data.R_rect_20,
                                                  trans_mtx=calib_data.T_cam2_velo)
        self.points: np.array = self.point_slices()

    def point_slices(self) -> np.array:
        velo = np.insert(self.points[:, 0:3], 3, 1, axis=1).T
        velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
        points_in_camera = self.camera_info.P.dot(self.camera_info.R.dot(self.camera_info.T.dot(velo)))
        points_in_camera[:2] /= points_in_camera[2, :]
        # filter point out of canvas
        u, v, z = points_in_camera
        u_out = np.logical_or(u < 0, u > self.image.width)
        v_out = np.logical_or(v < 0, v > self.image.height)
        outlier = np.logical_or(u_out, v_out)
        points_in_camera = np.delete(points_in_camera, np.where(outlier), axis=1).T
        velo = np.delete(velo, np.where(outlier), axis=1).T

        projection_list = np.array(points_in_camera[:, 0:2].astype(int))
        pts_coordinates = np.array(velo[:, 0:3])
        combined_data = np.hstack((pts_coordinates, projection_list))
        return np.array([tuple(row) for row in combined_data], dtype=point_dtype)

    def projection_test(self):
        # cam to velo: cam -> imu/ imu -> velo
        t_imu_velo = np.linalg.inv(self.c.T_velo_imu)
        t_cam_to_velo = np.dot(self.c.T_cam2_imu, t_imu_velo,)
        T = t_cam_to_velo
        lidar_pts = np.vstack((self.points['x'], self.points['y'], self.points['z'])).T
        lidar_pts = np.insert(lidar_pts[:, 0:3], 3, 1, axis=1).T
        P_new = np.hstack((self.camera_info.K, np.zeros((3, 1))))
        img_pts = P_new.dot(T.dot(lidar_pts))
        img_pts = img_pts / img_pts[2, :]
        img_pts = img_pts.T
        lidar_pts = lidar_pts.T
        projection_list = np.array(img_pts[:, 0:2].astype(int))
        pts_coordinates = np.array(lidar_pts[:, 0:3])
        combined_data = np.hstack((pts_coordinates, projection_list))
        return np.array([tuple(row) for row in combined_data], dtype=point_dtype)


class KittiDataLoader:
    """
    Class KittiDataLoader
    Loads the KITTI dataset from specified directory and provides methods to access the data.
    Attributes:
        name (str): Name of the dataset.
        data_dir (str): Directory path where the dataset is located.
        record_map (np.array): Numpy array containing the extracted date and drive information.
        first_only (bool): If True, only the first frame of each sequence will be returned.
        img_size (dict): Dictionary containing the width and height of the images.
        stereo_available (bool): Indicates if stereo data is available.
    Methods:
        __init__(self, data_dir, phase, first_only=False)
        __getitem__(self, idx)
        __len__(self)
        _read_kitty_data_structure(self)
    """
    def __init__(self, data_dir, phase, first_only=False):
        super().__init__()
        self.name: str = "KITTI-dataset"
        self.data_dir = os.path.join(data_dir, "kitti", phase)
        self.record_map: np.array = self._read_kitty_data_structure()
        self.first_only: bool = first_only
        self.img_size = {'width': 1242, 'height': 376}      # 1224 x 370 original
        self.stereo_available: bool = True
        print(f"Found {len(self.record_map)} Kitti record files.")

    def __getitem__(self, idx: int) -> List[KittiData]:
        scene = pykitti.raw(base_path=self.data_dir,
                            date=self.record_map[idx]['date'],
                            drive=self.record_map[idx]['drive'])
        kitti_data_chunk: List[KittiData] = []
        frame_num: int = 0
        for frame_idx in range(len(scene)):
            name = f"set_{str(idx)}_{self.record_map[idx]['date']}_{self.record_map[idx]['drive']}_{frame_num}"
            kitti_data_chunk.append(KittiData(name=name,
                                              rgb_stereo_pair=scene.get_rgb(frame_idx),
                                              velo_scan=scene.get_velo(frame_idx),
                                              calib_data=scene.calib))
            if self.first_only:
                break
            frame_num += 1
        return kitti_data_chunk

    def __len__(self) -> int:
        return len(self.record_map)

    def _read_kitty_data_structure(self) -> np.array:
        """
        Reads the kitty data structure from the specified directory.
        Args:
            self: The instance of the class.
        Returns:
            np.array: A numpy array containing the extracted date and drive information.
        """
        date_drive_list = []
        date_list = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        for date in date_list:
            drive_list = [name for name in os.listdir(os.path.join(self.data_dir, date)) if os.path.isdir(os.path.join(self.data_dir, date, name))]
            for drive in drive_list:
                date_drive_list.append({"date": date, "drive": drive.split("_")[4]})
        return date_drive_list
