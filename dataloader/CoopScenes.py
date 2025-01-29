import tensorflow as tf
import numpy as np
import open3d as o3d
import os
import glob
import yaml
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import patches
import aeifdataset as ad
from PIL import Image
from libraries.Stixel import point_dtype
from dataloader import BaseData, CameraInfo
from kiss_icp.preprocess import get_preprocessor
from kiss_icp.config import KISSConfig


def motion_compensation(points, timestamps, frame, last_frame):
    tf_last = ad.get_transformation(last_frame.vehicle.info)
    tf = ad.get_transformation(frame.vehicle.info)
    last_delta = tf_last.combine_transformation(tf.invert_transformation())

    k_config = KISSConfig()
    k_config.data.max_range = 200
    k_config.data.min_range = 0
    k_config.data.deskew = True
    k_config.registration.max_num_threads = 0
    preprocessor = get_preprocessor(k_config)

    lidar_points_compensated = preprocessor.preprocess(points, 1 - timestamps, last_delta.mtx)
    return lidar_points_compensated


def _get_timestamps(points):
    points_ts = points['t']
    normalized_points_ts = (points_ts - points_ts.min()) / (points_ts.max() - points_ts.min())
    return normalized_points_ts


class CoopSceneData(BaseData):
    def __init__(self, aeif_frame, record_name, view, last_frame=None):
        """
        Base class for raw from waymo open dataset
        Args:
            aeif_frame:
            label_only:
        """
        super().__init__()
        # self.camera_labels: open_dataset.CameraLabels = sorted(tf_frame.projected_lidar_labels, key=lambda i: i.name)[self.cam_idx]
        self.name: str = f"{record_name}_{aeif_frame.frame_id}_{view}"
        # self.laser_labels: Label = aeif_frame.laser_labels
        self.camera: ad.Camera = getattr(aeif_frame.vehicle.cameras, view)
        self.image: np.array = self.camera.image
        assert self.image.size == (1920, 1200), F"Check image Size of Camera:{view}"
        self.image_right = None     # Safety dummy, deprecated
        self.frame: ad.Frame = aeif_frame
        if view == "STEREO_LEFT":
            self.lidar: ad.Lidar = self.frame.vehicle.lidars.TOP
            """
            self.points = ad.combine_lidar_points(self.frame.vehicle.lidars.TOP,
                                                  self.frame.vehicle.lidars.LEFT,
                                                  self.frame.vehicle.lidars.RIGHT)
            """
        else:
            self.lidar: ad.Lidar = getattr(self.frame.tower.lidars, view)
        self.points = np.stack((self.lidar['x'], self.lidar['y'], self.lidar['z']), axis=-1)
        """ """
        if last_frame is not None:
            self.points = motion_compensation(points=self.points,
                                              timestamps=_get_timestamps(self.lidar),
                                              frame=self.frame,
                                              last_frame=last_frame)

        self.camera_info = CameraInfo(camera_mtx=self.camera.info.camera_mtx,
                                      trans_mtx=self._get_trans_mtx(),
                                      proj_mtx=self.camera.info.projection_mtx,
                                      rect_mtx=np.eye(4))

        self.camera_pose = self.camera.info.extrinsic[:,3][:3].reshape(3, 1)
        # Transformations
        self.points = self._point_slices()  # Apply laser transformation

    def _point_slices(self):
        points_3d = self.points # np.array([point.tolist()[:3] for point in self.lidar.points])
        points_3d = points_3d[points_3d[:, 0] <= 0]
        points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

        # Transform points to camera coordinates
        points_in_camera = self.camera_info.T.dot(points_3d_homogeneous.T).T

        # Apply rectification and projection to points
        points_in_camera = self.camera_info.R.dot(points_in_camera.T).T
        points_2d_homogeneous = self.camera_info.P.dot(points_in_camera.T)

        # Normalize by the third (z) component to get 2D image coordinates
        points_2d_homogeneous[:2] /= points_2d_homogeneous[2, :]

        u, v, z = points_2d_homogeneous
        u_out = np.logical_or(u < 0, u > self.image.width)
        v_out = np.logical_or(v < 0, v > self.image.height)
        outlier = np.logical_or(u_out, v_out)
        points_2d_homogeneous = np.delete(points_2d_homogeneous, np.where(outlier), axis=1).T
        velo = np.delete(points_3d_homogeneous.T, np.where(outlier), axis=1).T

        projection_list = np.array(points_2d_homogeneous[:, 0:3].astype(int))
        pts_coordinates = np.array(velo[:, 0:3])

        sem_seg = np.zeros(len(pts_coordinates))[:, np.newaxis]
        combined_data = np.hstack((pts_coordinates, projection_list, sem_seg))
        return np.array([tuple(row) for row in combined_data], dtype=point_dtype)

    def _get_trans_mtx(self):
        lidar_tf = ad.get_transformation(self.lidar)
        camera_tf = ad.get_transformation(self.camera)
        camera_inverse_tf = camera_tf.invert_transformation()
        return lidar_tf.combine_transformation(camera_inverse_tf).mtx


class CoopScenesDataLoader:
    def __init__(self, data_dir: str, phase: str, view='STEREO_LEFT', first_only=True):
        """
        Loads a full set of waymo raw in single frames, can be one tf_record file or a folder of  tf_record files.
        provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
        .images (5 views) and .laser_points (top lidar, divided into 5 fitting views). Like e.g.:
        798 tfrecord-files (selected by "idx")
            ~20 Frames (batch size/ dataset - selected by "frame_num")
                5 .images (camera view - selected by index[])
                5 .laser_points (shape of [..., [x, y, z, img_x, img_y]])
        Args:
            data_dir: specify the location of the tf_records
            additional_views: if True, loads only frames with available camera segmentation
            first_only: doesn't load the full ~20 frames to return a raw sample if True
        """
        super().__init__()
        self.name: str = "coopscenes"
        self.phase: str = phase
        self.data_dir = os.path.join(data_dir, "coopscenes", phase)
        self.first_only: bool = first_only
        self.img_size = {'width': 1920, 'height': 1200}
        with open(f'dataloader/configs/{self.name}-pcl-config.yaml') as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.stereo_available: bool = False
        self.view = view
        # find files in folder (data_dir) which fit to pattern endswith-'.tfrecord'
        self.record_map = sorted(glob.glob(os.path.join(self.data_dir, '**', '*.4mse'), recursive=True))
        print(f"Found {len(self.record_map)} CoopScenes files")

    def __getitem__(self, idx):
        record = ad.DataRecord(self.record_map[idx])
        aeif_data_chunk = []
        last_frame = None
        for frame_num, aeif_frame in enumerate(record):
            # start_time = datetime.now()
            if frame_num % 5 == 0:
                if self.view == "VIEW":
                    for i in range(1, 3):
                        aeif_data_chunk.append(CoopSceneData(aeif_frame=aeif_frame,
                                                             record_name=record.name,
                                                             view=f"{self.view}_{i}"))
                else:
                    # View = STEREO_LEFT
                    aeif_data_chunk.append(CoopSceneData(aeif_frame=aeif_frame,
                                                         record_name=record.name,
                                                         view=self.view,
                                                         last_frame=last_frame))
                # self.object_creation_time = datetime.now() - start_time
                if self.first_only:
                    break
            last_frame = aeif_frame
        return aeif_data_chunk

    def __len__(self):
        return len(self.record_map)
