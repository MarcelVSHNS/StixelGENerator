import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from PIL import Image


class WaymoDataLoader(object):
    def __init__(self, data_dir, tfrecord_file=None, camera_segmentation_only=False, shuffle=False):
        """
        Loads a full set of waymo data in single frames, can be one tf_record file or a folder of  tf_record files.
        Args:
            data_dir: specify the location of the tf_records
            tf_record_file: pick a specific file from data_dir
            camera_segmentation_only: if True, loads only frames with available camera segmentation
            shuffle: if True, the frames will be shuffled
        """
        self.data_dir = data_dir
        self.camera_segmentation_only = camera_segmentation_only
        self.frames = []
        if tfrecord_file:
            self.tfrecord_file = os.path.join(data_dir, tfrecord_file)
            assert self.tfrecord_file.endswith('.tfrecord')
            self.unpack_single_tfrecord_file_from_path(self.tfrecord_file)
        else:
            self.tfrecord_file = data_dir
            self.load_data_from_folder()
        if shuffle:
            random.shuffle(self.frames)
        print(f"Num_frames: {len(self.frames)}")

    def load_data_from_folder(self):
        data_folder = os.path.join(self.data_dir, '*.tfrecord')
        dataset_list = glob.glob(data_folder)
        print(f"Found {len(dataset_list)} record files")
        for dataset_name in dataset_list:
            # unpack every used frame from set and store in frame list
            self.unpack_single_tfrecord_file_from_path(dataset_name)

    def unpack_single_tfrecord_file_from_path(self, tf_record_filename):
        """ Loads a tf-record file from the given path and returns a list of frames from the file """
        dataset = tf.data.TFRecordDataset(tf_record_filename, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if self.camera_segmentation_only:
                if frame.images[0].camera_segmentation_label.panoptic_label:
                    self.frames.append(WaymoData(frame))
            else:
                self.frames.append(WaymoData(frame))
            # if self.laser_segmentation_only:
                # if frame.lasers[0].ri_return1.segmentation_label_compressed:
                #    frame_list.append(frame)


class WaymoData(object):
    def __init__(self, tf_frame, camera_segmentation_only=False):
        # Base declaration
        self.images = []
        self.laser_points = []
        self.laser_camera_projections = []
        self.camera_segmentation_only = camera_segmentation_only
        # Transformations
        self.top_lidar_points_slices(tf_frame)      # Apply laser transformation
        self.convert_images_to_pil(tf_frame.images)                # Apply image transformation
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
            self.laser_points.append(tf.gather_nd(laser_points, tf.where(mask)).numpy())
            self.laser_camera_projections.append(tf.cast(tf.gather_nd(laser_projection_points, tf.where(mask)), dtype=tf.float32).numpy())

    def convert_images_to_pil(self, frame_images):
        images = sorted(frame_images, key=lambda i: i.name)
        for image in images:
            self.images.append(Image.fromarray(tf.image.decode_jpeg(image.image).numpy()))
