import tensorflow as tf
import numpy as np
import os
import glob
import open3d as o3d
from datetime import datetime

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from PIL import Image


class WaymoDataLoader:
    def __init__(self, data_dir, camera_segmentation_only=False, first_only=False):
        """
        Loads a full set of waymo data in single frames, can be one tf_record file or a folder of  tf_record files.
        provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
        .images (5 views) and .laser_points (top lidar, divided into 5 fitting views). Like e.g.:
        798 tfrecord-files (selected by "idx")
            ~20 Frames (batch size/ dataset - selected by "frame_num")
                5 .images (camera view - selected by index[])
                5 .laser_points (shape of [..., [x, y, z, img_x, img_y]])
        Args:
            data_dir: specify the location of the tf_records
            camera_segmentation_only: if True, loads only frames with available camera segmentation
            first_only: doesn't load the full ~20 frames to return a data sample if True
        """
        self.data_dir = data_dir
        self.camera_segmentation_only = camera_segmentation_only
        self.first_only = first_only
        # find files in folder (data_dir) which fit to pattern endswith-'.tfrecord'
        self.tfrecord_map = glob.glob(os.path.join(self.data_dir, '*.tfrecord'))
        print(f"Found {len(self.tfrecord_map)} record files")

    def __getitem__(self, idx):
        frames = self.unpack_single_tfrecord_file_from_path(self.tfrecord_map[idx])
        waymo_data_chunk = []
        for tf_frame in frames:
            # start_time = datetime.now()
            waymo_data_chunk.append(WaymoData(tf_frame, camera_segmentation_only=self.camera_segmentation_only))
            # self.object_creation_time = datetime.now() - start_time
            if self.first_only:
                break
        return waymo_data_chunk

    def __len__(self):
        return len(self.tfrecord_map)

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
                if frame_num % 50 == 0:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    frame_list.append(frame)
                frame_num += 1
        return frame_list


class WaymoData:
    def __init__(self, tf_frame, camera_segmentation_only=False):
        """
        Base class for data from waymo open dataset
        Args:
            tf_frame:
            camera_segmentation_only:
        """
        # Base declaration
        self.images = []
        self.laser_points = []
        self.camera_segmentation_only = camera_segmentation_only
        self.name = tf_frame.context.name
        # Transformations
        self.top_lidar_points_slices(tf_frame)  # Apply laser transformation
        self.convert_images_to_pil(tf_frame.images)  # Apply image transformation
        # TODO: Future Add-On
        if self.camera_segmentation_only:
            self.camera_labels, self.camera_instance_labels = None, None

    def top_lidar_points_slices(self, frame):
        """
        Slices the top lidar point cloud into pieces, fitting for every camera incl. projections
        Args:
            frame: Expects the full frame
        Returns: A list of points and a list of projections: (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
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
        # cp_points_all_tensor[..., 0] while 0 means cameraName.name (first projection)
        for view in range(len(images)):
            # Create mask from image
            mask = tf.equal(laser_projection_points[..., 0], images[view].name)
            # transform points after slicing it from the mask into float values
            laser_points_view = tf.gather_nd(laser_points, tf.where(mask)).numpy()
            laser_camera_projections_view = tf.cast(tf.gather_nd(laser_projection_points, tf.where(mask)), dtype=tf.float32).numpy()
            concatenated_laser_pts = np.column_stack((laser_points_view, laser_camera_projections_view[..., 1:3]))
            self.laser_points.append(concatenated_laser_pts)

    def convert_images_to_pil(self, frame_images):
        images = sorted(frame_images, key=lambda i: i.name)
        for image in images:
            self.images.append(Image.fromarray(tf.image.decode_jpeg(image.image).numpy()))
