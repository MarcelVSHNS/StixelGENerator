import tensorflow as tf
import numpy as np
import os
import glob
import yaml
from typing import List, Tuple
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, transform_utils
from waymo_open_dataset.v2 import convert_range_image_to_point_cloud
from PIL import Image
from libraries.Stixel import point_dtype
from dataloader import BaseData, CameraInfo


class WaymoData(BaseData):
    def __init__(self, tf_frame, camera_segmentation_only=False):
        """
        Base class for raw from waymo open dataset
        Args:
            tf_frame:
            camera_segmentation_only:
        """
        super().__init__()
        self.name: str = tf_frame.context.name

        front_img = sorted(tf_frame.images, key=lambda i: i.name)[0]
        self.image = Image.fromarray(tf.image.decode_jpeg(front_img.image).numpy())     # Front image
        self.image_right = None     # Safety dummy, deprecated
        self.camera_segmentation_only: bool = camera_segmentation_only
        self.frame: open_dataset.Frame = tf_frame
        front_cam_calib = sorted(self.frame.context.camera_calibrations, key=lambda i: i.name)[0]
        P = self._get_projection_matrix(front_cam_calib.intrinsic)
        T = np.linalg.inv(np.array(front_cam_calib.extrinsic.transform).reshape(4, 4))
        self.camera_info = CameraInfo(camera_mtx=P[:3, :3],
                                      trans_mtx=T,
                                      proj_mtx=P,
                                      rect_mtx=np.eye(4))
        # Transformations
        self.point_slices(tf_frame)  # Apply laser transformation

    @staticmethod
    def _get_projection_matrix(intrinsics):
        # Projection matrix: https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        # Extract intrinsic parameters: 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
        waymo_cam_RT = np.array([0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)
        f_u, f_v, c_u, c_v = intrinsics[:4]
        # Construct the camera matrix K
        K_tmp = np.array([
            [f_u, 0, c_u, 0],
            [0, f_v, c_v, 0],
            [0, 0, 1, 0]
        ])
        return K_tmp @ waymo_cam_RT

    def point_slices(self, frame):
        """
        Slices the top lidar point cloud into pieces, fitting for every camera incl. projections
        Args:
            frame: Expects the full frame
        Returns: A list of points and a list of projections: (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
        shape: [x, y, z, proj_x, proj_y]
        """
        # Cuts for just the front view (front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4)
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        # 3d points in vehicle frame, just pick Top LiDAR
        laser_points = points[0]
        laser_projection_points = tf.constant(cp_points[0], dtype=tf.int32)
        images = sorted(frame.images, key=lambda i: i.name)
        # define mask where the projections equal the picture view
        # (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
        mask = tf.equal(laser_projection_points[..., 0], images[0].name)
        # transform points after slicing it from the mask into float values
        laser_points_view = tf.gather_nd(laser_points, tf.where(mask)).numpy()
        laser_camera_projections_view = tf.cast(tf.gather_nd(laser_projection_points, tf.where(mask)), dtype=tf.float32).numpy()
        concatenated_laser_pts = np.column_stack((laser_points_view, laser_camera_projections_view[..., 1:3]))
        self.points = np.array([tuple(row) for row in concatenated_laser_pts], dtype=point_dtype)

    def projection_test(self):
        lidar_pts = np.vstack((self.points['x'], self.points['y'], self.points['z'])).T
        lidar_pts = np.insert(lidar_pts[:, 0:3], 3, 1, axis=1).T
        img_pts = self.camera_info.P.dot(self.camera_info.R.dot(self.camera_info.T.dot(lidar_pts)))
        img_pts[:2] /= img_pts[2, :]
        img_pts = img_pts.T
        lidar_pts = lidar_pts.T
        projection_list = np.array(img_pts[:, 0:2].astype(int))
        pts_coordinates = np.array(lidar_pts[:, 0:3])
        combined_data = np.hstack((pts_coordinates, projection_list))
        return np.array([tuple(row) for row in combined_data], dtype=point_dtype)


class WaymoDataLoader:
    def __init__(self, data_dir: str, phase: str, camera_segmentation_only=False, first_only=True):
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
            camera_segmentation_only: if True, loads only frames with available camera segmentation
            first_only: doesn't load the full ~20 frames to return a raw sample if True
        """
        super().__init__()
        self.name: str = "waymo-od"
        self.phase: str = phase
        self.data_dir = os.path.join(data_dir, "waymo", phase)
        self.camera_segmentation_only: bool = camera_segmentation_only
        self.first_only: bool = first_only
        self.img_size = {'width': 1920, 'height': 1280}
        with open(f'dataloader/configs/{self.name}-pcl-config.yaml') as yaml_file:
            self.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.stereo_available: bool = False
        # find files in folder (data_dir) which fit to pattern endswith-'.tfrecord'
        self.record_map = sorted(glob.glob(os.path.join(self.data_dir, '*.tfrecord')))
        print(f"Found {len(self.record_map)} tf record files")

    def __getitem__(self, idx):
        try:
            frames = self.unpack_single_tfrecord_file_from_path(self.record_map[idx])
        except Exception as e:
            with open(f"failures_{self.phase}.txt", "a") as file:
                file_name = os.path.basename(self.record_map[idx])
                file.write(f"  gs://waymo_open_dataset_v_1_4_3/individual_files/{self.phase}/{file_name} \\ \n")
            print(f"Fail to open {file_name}, documented.")
            raise e
        waymo_data_chunk = []
        for tf_frame in frames:
            # start_time = datetime.now()
            waymo_data_chunk.append(WaymoData(tf_frame=tf_frame,
                                              camera_segmentation_only=self.camera_segmentation_only))
            # self.object_creation_time = datetime.now() - start_time
            if self.first_only:
                break
        return waymo_data_chunk

    def __len__(self):
        return len(self.record_map)

    def unpack_single_tfrecord_file_from_path(self, tf_record_filename):
        """ Loads a tf-record file from the given path. Picks only every tenth frame to reduce the dataset and increase
        the diversity of it. With camera_segmentation_only = True, every availabe frame is picked (every tenth is annotated)
        Args:
            tf_record_filename: full path and name of the tf_record file to open
        Returns: a list of frames from the file
        """
        dataset = tf.data.TFRecordDataset(tf_record_filename, compression_type='')
        frame_list = []
        frame_num = 0
        for data in dataset:
            if self.camera_segmentation_only:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                if frame.images[0].camera_segmentation_label.panoptic_label:
                    frame_list.append(frame)
            else:
                if frame_num % 10 == 0:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    frame_list.append(frame)
                frame_num += 1
        return frame_list
