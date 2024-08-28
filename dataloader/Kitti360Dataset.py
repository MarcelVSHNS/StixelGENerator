import numpy as np
import pandas as pd
import pykitti
import os
import yaml
from typing import List, Tuple, TypedDict, Optional
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid
from kitti360scripts.helpers.annotation import Annotation3D, global2local
from libraries.Stixel import point_dtype
from dataloader import BaseData, CameraInfo, Pose
from PIL import Image
import open3d


def _parse_calibration_file(file_path):
    calib_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            calib_data[key.strip()] = value.strip()
    return calib_data


class Kitti360Data(BaseData):
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
    def __init__(self,
                 name: str,
                 image: Image.Image,
                 velo_scan: np.array,
                 calib_data,
                 image_right: Optional[Image.Image] = None):
        super().__init__()
        self.name: str = name
        self.image = image     # cam2, rectified
        self.image_right = image_right       # cam3, rectified
        self.points: np.array = velo_scan
        self.c = calib_data
        self.camera_info: CameraInfo = CameraInfo(camera_mtx=calib_data["K_cam0"],
                                                  proj_mtx=calib_data["P_rect_cam0"],
                                                  rect_mtx=np.eye(4),
                                                  trans_mtx=calib_data["T_sick2cam0"])
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
        pts = np.array([tuple(row) for row in combined_data], dtype=point_dtype)
        return pts

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
        pts = np.array([tuple(row) for row in combined_data], dtype=point_dtype)
        return pts


class Kitti360DataLoader:
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
        self.name: str = "kitti-360"
        self.phase: str = phase
        self.data_dir = os.path.join(data_dir, "KITTI-360")
        self.record_map: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, f"{phase}_files_map.csv"), header=None, names=["Drive", "Frame"])
        with open(f'dataloader/configs/{self.name}-pcl-config.yaml') as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.first_only: bool = first_only
        self.calib_data = self._load_calib()
        self.img_size = {'width': self.calib_data["S_rect_cam0"][0], 'height': self.calib_data["S_rect_cam0"][1]}
        self.stereo_available: bool = False
        print(f"Found {len(self.record_map)} Kitti-360 files.")

    def __getitem__(self, idx: int) -> List[Kitti360Data]:
        name = f"{self.record_map.loc[idx]['Drive']}-{os.path.splitext(self.record_map.loc[idx]['Frame'])[0]}"
        velo_scan, sick_scan = self._read_3d_pts(idx_df=self.record_map.iloc[idx])
        image = self._read_2d_image(idx_df=self.record_map.iloc[idx])
        #self._read_bbox(idx_df=self.record_map.iloc[idx])
        kitti360_data = Kitti360Data(name=name,
                                     image=image,
                                     velo_scan=sick_scan,
                                     calib_data=self.calib_data)
        return [kitti360_data]

    def __len__(self) -> int:
        return len(self.record_map)

    def _read_2d_image(self, idx_df: pd.DataFrame) -> Image.Image:
        """   """
        data_2d_raw_path = os.path.join(self.data_dir, "data_2d_raw", idx_df.loc["Drive"], "image_00",
                                        "data_rect", idx_df.loc["Frame"])
        return Image.open(data_2d_raw_path)

    def _read_3d_pts(self, idx_df: pd.DataFrame, sick_offset: int = 15) -> Tuple[np.array, np.array]:
        """   """
        data_3d_velo_path = os.path.join(self.data_dir, "data_3d_raw", idx_df.loc["Drive"], "velodyne_points",
                                         "data", os.path.splitext(idx_df.loc["Frame"])[0] + ".bin")
        # file has always 10 chars
        frame = int(os.path.splitext(idx_df.loc["Frame"])[0]) * 4 + sick_offset

        # Velodyne Scans: x,y,z,intensity
        velo_scan = np.fromfile(data_3d_velo_path, dtype=np.float32)
        velo_scan = velo_scan.reshape((-1, 4))

        # Sick scans: x,y,z WITHOUT Intensity. All measurements are below the horizon.
        # TODO: select by timestamp, data are inconsistent.
        sick_scans = []
        for scan in range(frame, frame + 1000):
            data_3d_sick_path = os.path.join(self.data_dir, "data_3d_raw", idx_df.loc["Drive"], "sick_points",
                                             "data", str(scan).zfill(10) + ".bin")
            sick_scan = np.fromfile(data_3d_sick_path, dtype=np.float32)
            sick_scan = np.reshape(sick_scan,[-1,2])
            sick_scan = np.concatenate([np.zeros_like(sick_scan[:,0:1]), -sick_scan[:,0:1], sick_scan[:,1:2]], axis=1)
            sick_scans.append(sick_scan)
        sick_scans = np.concatenate(sick_scans, axis=0)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(sick_scans[:, :3])
        open3d.visualization.draw_geometries([pcd])
        return velo_scan, sick_scans

    def _read_bbox(self, idx_df: pd.DataFrame):
        bbox_path = os.path.join(self.data_dir, "data_3d_bboxes")
        annotation3d = Annotation3D(bbox_path, idx_df.loc["Drive"])
        return None

    def _load_calib(self):
        """ cam is reference with 0,0,0 0,0,0. Provides camera intrinsics (and velo/sick extrinsic) """
        cam2velo = loadCalibrationRigid(os.path.join(self.data_dir, 'calibration', 'calib_cam_to_velo.txt'))
        sick2velo = loadCalibrationRigid(os.path.join(self.data_dir, 'calibration', 'calib_sick_to_velo.txt'))
        intrinsics = _parse_calibration_file(os.path.join(self.data_dir, 'calibration', 'perspective.txt'))
        T_velo2cam = np.linalg.inv(cam2velo)
        # cam to sick: cam -> velo / velo -> sick
        t_velo_sick = np.linalg.inv(sick2velo)
        T_sick2cam = cam2velo @ t_velo_sick
        # T_sick2cam = T_velo2cam @ sick2velo
        # camera mtx rgb_left
        K_cam0 = np.array(intrinsics['K_00'].split(), dtype=np.float32).reshape(3, 3)
        P_rect_cam0 = np.array(intrinsics['P_rect_00'].split(), dtype=np.float32).reshape(3, 4)
        R_rect_cam0 = np.array(intrinsics['R_rect_00'].split(), dtype=np.float32).reshape(3, 3)
        S_rect_cam0 = np.array(intrinsics['S_rect_00'].split(), dtype=np.float32)
        calib_data = {"T_velo2cam0": T_velo2cam,
                      "T_sick2cam0": T_sick2cam,
                      "K_cam0": K_cam0,
                      "P_rect_cam0": P_rect_cam0,
                      "S_rect_cam0": S_rect_cam0,
                      "R_rect_cam0": R_rect_cam0}
        return calib_data
