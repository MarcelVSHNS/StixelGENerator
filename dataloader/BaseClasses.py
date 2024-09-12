import numpy as np
from typing import Optional, List, Dict, TypedDict
from PIL import Image
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    def __init__(self):
        self.name: Optional[str] = None
        self.data_dir: Optional[str] = None
        self.record_map: Optional[List[str]] = None
        self.img_size: ImgSize = {'width': 0, 'height': 0}
        self.first_only: bool = False
        self.stereo_available: bool = False

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass


class BaseData(ABC):
    def __init__(self):
        self.name: Optional[str] = None
        self.image: Optional[Image] = None
        self.points: Optional[np.array] = None
        self.camera_info: Optional[CameraInfo] = None


class CameraInfo:
    """
    Class to store camera information.
    Attributes:
        extrinsic (Pose): The extrinsic pose of the camera.
        K (np.array): The camera matrix.
        P (np.array): The projection matrix.
        R (np.array): The rectification matrix.
    Methods:
        __init__(self, xyz: np.array, rpy: np.array, camera_mtx: np.array, projection_mtx: np.array,
            rectification_mtx: np.array):
            Initializes the CameraInformation object with the given camera information.
    """
    def __init__(self, camera_mtx: np.array, trans_mtx: np.array, proj_mtx: np.array, rect_mtx: np.array):
        self.K = camera_mtx
        self.T = trans_mtx
        self.P = proj_mtx
        self.R = rect_mtx
        self.extrinsic: Pose = Pose(xyz=self.T[:3, 3],
                                    rpy=self.calc_euler_angle_from_trans_mtx(self.T))
        self.D = np.array([])

    @staticmethod
    def calc_euler_angle_from_trans_mtx(trans_mtx: np.array):
        """
        Returns the euler angle (roll, pitch, yaw) of the camera matrix from the given transformation matrix.
        """
        rota_mtx = trans_mtx[:3, :3]  # rotation matrix
        yaw = np.arctan2(rota_mtx[1, 0], rota_mtx[0, 0])
        pitch = np.arctan2(-rota_mtx[2, 0], np.sqrt(rota_mtx[2, 1] ** 2 + rota_mtx[2, 2] ** 2))
        roll = np.arctan2(rota_mtx[2, 1], rota_mtx[2, 2])
        return np.array([roll, pitch, yaw])


class ImgSize(TypedDict):
    """
    A Dict representing the size of an image.
    Attributes:
        width (int): The width of the image.
        height (int): The height of the image.
    """
    width: int
    height: int


class Pose:
    """
    Initializes a new Pose object.
    Args:
        xyz (np.array): The position vector in x, y, and z coordinates.
        rpy (np.array): The orientation vector in roll, pitch, and yaw angles.
    """
    def __init__(self, xyz: np.array, rpy: np.array):
        self.xyz: np.array = xyz
        self.rpy: np.array = rpy
