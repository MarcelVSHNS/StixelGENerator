import numpy as np
import pykitti
import os
from typing import List, Tuple

from libraries.names import point_dtype


class CameraInformation:
    def __init__(self, xyz: np.array, rpy: np.array, camera_mtx: np.array, projection_mtx: np.array, rectification_mtx: np.array):
        self.extrinsic: Pose = Pose(xyz, rpy)
        self.camera_mtx: np.array = camera_mtx
        self.projection_mtx: np.array = projection_mtx
        self.rectification_mtx: np.array = rectification_mtx


class Pose:
    def __init__(self, xyz: np.array, rpy: np.array):
        self.xyz: np.array = xyz
        self.rpy: np.array = rpy


class KittiData:
    def __init__(self, name, rgb_stereo_pair, velo_scan, calib_data):
        self.name: str = name
        self.image = rgb_stereo_pair[0]     # cam2, rectified
        self.image_right = rgb_stereo_pair[1]       # cam3, rectified
        self.points: np.array = velo_scan
        self.t_cam2_velo = calib_data.T_cam2_velo       # transformation matrix
        rota_mtx = self.t_cam2_velo[:3, :3]  # Die Rotationsmatrix
        yaw = np.arctan2(rota_mtx[1, 0], rota_mtx[0, 0])
        pitch = np.arctan2(-rota_mtx[2, 0], np.sqrt(rota_mtx[2, 1] ** 2 + rota_mtx[2, 2] ** 2))
        roll = np.arctan2(rota_mtx[2, 1], rota_mtx[2, 2])
        self.camera_info: CameraInformation = CameraInformation(xyz=self.t_cam2_velo[:3, 3],
                                                                rpy=np.array([roll, pitch, yaw]),
                                                                camera_mtx=calib_data.K_cam2,
                                                                projection_mtx=np.array(calib_data.P_rect_20),
                                                                rectification_mtx=calib_data.R_rect_20)
        self.points: np.array = self.point_slices()

    def point_slices(self) -> np.array:
        velo = np.insert(self.points[:, 0:3], 3, 1, axis=1).T
        velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
        points_in_camera = self.camera_info.projection_mtx.dot(self.camera_info.rectification_mtx.dot(self.t_cam2_velo.dot(velo)))
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

class KittiDataLoader:
    def __init__(self, data_dir, phase, first_only=False):
        self.name: str = "KITTI-dataset"
        self.data_dir = os.path.join(data_dir, "kitti", phase)
        self.kitti_record_map: np.array = self._read_kitty_data_structure()
        self.first_only: bool = first_only
        self.img_size = {'width': 1242, 'height': 376}      # 1224 x 370 original
        self.stereo_available: bool = True
        print(f"Found {len(self.kitti_record_map)} Kitti record files.")

    def __getitem__(self, idx: int) -> List[KittiData]:
        scene = pykitti.raw(base_path=self.data_dir,
                             date=self.kitti_record_map[idx]['date'],
                             drive=self.kitti_record_map[idx]['drive'])

        kitti_data_chunk: List[KittiData] = []
        frame_num: int = 0
        for frame_idx in range(len(scene)):
            name = f"set_{str(idx)}_{self.kitti_record_map[idx]['date']}_{self.kitti_record_map[idx]['drive']}_{frame_num}"
            kitti_data_chunk.append(KittiData(name=name,
                                              rgb_stereo_pair=scene.get_rgb(frame_idx),
                                              velo_scan=scene.get_velo(frame_idx),
                                              calib_data=scene.calib))
            if self.first_only:
                break
            frame_num += 1
        return kitti_data_chunk

    def __len__(self) -> int:
        return len(self.kitti_record_map)

    def _read_kitty_data_structure(self) -> np.array:
        date_drive_list = []
        date_list = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        for date in date_list:
            drive_list = [name for name in os.listdir(os.path.join(self.data_dir, date)) if os.path.isdir(os.path.join(self.data_dir, date, name))]
            for drive in drive_list:
                date_drive_list.append({"date": date, "drive": drive.split("_")[4]})
        return date_drive_list
