import tensorflow as tf
import numpy as np
import os
import glob
import yaml
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import patches
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.label_pb2 import Label
from waymo_open_dataset.utils import frame_utils, transform_utils, box_utils
from waymo_open_dataset.v2 import convert_range_image_to_point_cloud
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from PIL import Image
from libraries.Stixel import point_dtype
#from libraries import draw_3d_wireframe_box, draw_2d_box
from dataloader import BaseData, CameraInfo


def show_camera_image(camera_image, layout):
  """Display the given camera image."""
  ax = plt.subplot(*layout)
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')
  return ax


def show_projected_lidar_labels(frame, camera_image, ax):
  """Displays pre-projected 3D laser labels."""

  for projected_labels in frame.projected_lidar_labels:
    # Ignore camera labels that do not correspond to this camera.
    if projected_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in projected_labels.labels:
      # Draw the bounding box.
      rect = patches.Rectangle(
          xy=(label.box.center_x - 0.5 * label.box.length,
              label.box.center_y - 0.5 * label.box.width),
          width=label.box.length,
          height=label.box.width,
          linewidth=1,
          edgecolor=(0.0, 1.0, 0.0, 1.0),  # green
          facecolor=(0.0, 1.0, 0.0, 0.1))  # opaque green
      ax.add_patch(rect)


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.
    FROM: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_3d_semseg.ipynb
    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

"""
def show_projected_camera_synced_boxes(frame, camera_image, ax, draw_3d_box=False):
  # Displays camera_synced_box 3D labels projected onto camera.
  FILTER_AVAILABLE = any(
      [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels])

  if not FILTER_AVAILABLE:
      print('WARNING: num_top_lidar_points_in_box does not seem to be populated. '
            'Make sure that you are using an up-to-date release (V1.3.2 or later) '
            'to enable improved filtering of occluded objects.')

  # Fetch matching camera calibration.
  calibration = next(cc for cc in frame.context.camera_calibrations
                     if cc.name == camera_image.name)

  for label in frame.laser_labels:
    box = label.camera_synced_box

    if not box.ByteSize():
      continue  # Filter out labels that do not have a camera_synced_box.
    if (FILTER_AVAILABLE and not label.num_top_lidar_points_in_box) or (
        not FILTER_AVAILABLE and not label.num_lidar_points_in_box):
      continue  # Filter out likely occluded objects.

    # Retrieve upright 3D box corners.
    box_coords = np.array([[
        box.center_x, box.center_y, box.center_z, box.length, box.width,
        box.height, box.heading
    ]])
    corners = box_utils.get_upright_3d_box_corners(
        box_coords)[0].numpy()  # [8, 3]

    # Project box corners from vehicle coordinates onto the image.
    projected_corners = project_vehicle_to_image(frame.pose, calibration,
                                                 corners)
    u, v, ok = projected_corners.transpose()
    ok = ok.astype(bool)

    # Skip object if any corner projection failed. Note that this is very
    # strict and can lead to exclusion of some partially visible objects.
    if not all(ok):
      continue
    u = u[ok]
    v = v[ok]

    # Clip box to image bounds.
    u = np.clip(u, 0, calibration.width)
    v = np.clip(v, 0, calibration.height)

    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
      continue

    if draw_3d_box:
      # Draw approximate 3D wireframe box onto the image. Occlusions are not
      # handled properly.
      draw_3d_wireframe_box(ax, u, v, (1.0, 1.0, 0.0))
    else:
      # Draw projected 2D box onto the image.
      draw_2d_box(ax, u, v, (1.0, 1.0, 0.0))
"""

class WaymoData(BaseData):
    def __init__(self, tf_frame, frame_num, cam_idx):
        """
        Base class for raw from waymo open dataset
        Args:
            tf_frame:
            label_only:
        """
        super().__init__()
        # front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4
        self.cam_idx: int = cam_idx
        img = sorted(tf_frame.images, key=lambda i: i.name)[cam_idx]
        # self.camera_labels: open_dataset.CameraLabels = sorted(tf_frame.projected_lidar_labels, key=lambda i: i.name)[self.cam_idx]
        self.name: str = f"{tf_frame.context.name}_{frame_num}_{open_dataset.CameraName.Name.Name(img.name)}"
        self.laser_labels: Label = tf_frame.laser_labels
        self.image: np.array = Image.fromarray(tf.image.decode_jpeg(img.image).numpy())
        assert self.image.size == (1920, 1280), F"Check image Size of Camera idx:{self.cam_idx}"
        self.image_right = None     # Safety dummy, deprecated
        self.frame: open_dataset.Frame = tf_frame
        front_cam_calib = sorted(self.frame.context.camera_calibrations, key=lambda i: i.name)[0]
        P = self._get_projection_matrix(front_cam_calib.intrinsic)
        T = np.linalg.inv(np.array(front_cam_calib.extrinsic.transform).reshape(4, 4))
        self.camera_info = CameraInfo(camera_mtx=P[:3, :3],
                                      trans_mtx=T,
                                      proj_mtx=P,
                                      rect_mtx=np.eye(4))
        self.camera_info.D = np.array(front_cam_calib.intrinsic[4:])
        # Transformations
        self._point_slices(frame=tf_frame, cam_idx=self.cam_idx)  # Apply laser transformation

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

    def _point_slices(self, frame, cam_idx):
        """
        Slices the top lidar point cloud into pieces, fitting for every camera incl. projections
        Args:
            frame: Expects the full frame
        Returns: A list of points and a list of projections: (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
        shape: [x, y, z, u, proj_y]
        """
        # Cuts for just the front view (front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4)
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        point_labels = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels)
        self.all_points = points[0]
        laser_labels = point_labels[0]
        self.laser_projection_points = tf.constant(cp_points[0], dtype=tf.int32)
        images = sorted(frame.images, key=lambda i: i.name)
        self.mask = tf.equal(self.laser_projection_points[..., 0], images[cam_idx].name)
        # transform points after slicing it from the mask into float values
        laser_points_view = tf.gather_nd(self.all_points, tf.where(self.mask)).numpy()
        laser_labels_view = tf.gather_nd(laser_labels, tf.where(self.mask)).numpy()
        # laser_camera_projections_view = tf.cast(tf.gather_nd(laser_projection_points, tf.where(mask)), dtype=tf.float32).numpy()
        # concatenated_laser_pts = np.column_stack((laser_points_view, laser_camera_projections_view[..., 1:3], laser_labels_view[..., 1:]))
        # self.points = np.array([tuple(row) for row in concatenated_laser_pts], dtype=point_dtype)
        self.projection = self._point_projection(laser_points_view)
        combined_data = np.hstack((laser_points_view, self.projection, laser_labels_view[..., 1:]))
        # TODO: drop the waymo projection mask and use the projection (enables left and right image)
        width, height = self.image.size
        valid_indices = (
                (combined_data[:, 3] >= 0) & (combined_data[:, 3] < width) &
                (combined_data[:, 4] >= 0) & (combined_data[:, 4] < height)
        )
        valid_combined_data = combined_data[valid_indices]
        self.points = np.array([tuple(row) for row in valid_combined_data], dtype=point_dtype)

    def _point_projection(self, points):
        lidar_pts = np.insert(points[:, 0:3], 3, 1, axis=1).T
        img_pts = self.camera_info.P.dot(self.camera_info.R.dot(self.camera_info.T.dot(lidar_pts)))

        img_pts[:2] /= img_pts[2, :]
        img_pts = img_pts.T
        return img_pts

    def inverse_projection(self, points):
        img_pts = np.vstack((points['u'], points['v'], points['w'])).T
        img_pts = np.insert(img_pts[:, 0:3], 3, 1, axis=1).T
        img_pts[:2] *= img_pts[2, :]
        # Homogeneous coordinates in the image plane
        k_exp = np.eye(4)
        k_exp[:3, :3] = self.camera_info.K
        T_inverted_x = self.camera_info.T.copy()
        # T_inverted_x[0, 3] *= 0
        # T_inverted_x[1, 3] *= 0
        P = k_exp @ T_inverted_x
        # (K * T)-1 * pt
        lidar_pts = np.linalg.inv(P) @ img_pts
        lidar_pts = lidar_pts.T
        pts_coordinates = np.array(lidar_pts[:, 0:3])
        # Convert from homogeneous coordinates to 3D (divide by the last element)
        return pts_coordinates
    """
    def show_3d_bboxes(self):
        for index, image in enumerate(self.frame.images):
            ax = show_camera_image(image, [3, 3, index + 1])
            show_projected_lidar_labels(self.frame, image, ax)
            show_projected_camera_synced_boxes(self.frame, image, ax, draw_3d_box=False)
        plt.show()
    """


class WaymoDataLoader:
    def __init__(self, data_dir: str, phase: str, additional_views=False, first_only=True):
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
        self.name: str = "waymo-od"
        self.phase: str = phase
        self.data_dir = os.path.join(data_dir, "waymo", phase)
        self.additional_views: bool = additional_views
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
        for frame_num, tf_frame in frames.items():
            # start_time = datetime.now()
            if self.additional_views:
                for i in range(3):
                    waymo_data_chunk.append(WaymoData(tf_frame=tf_frame,
                                                      frame_num=frame_num,
                                                      cam_idx=i))
            else:
                waymo_data_chunk.append(WaymoData(tf_frame=tf_frame,
                                                  frame_num=frame_num,
                                                  cam_idx=0))
                # self.object_creation_time = datetime.now() - start_time
            if self.first_only:
                break
        return waymo_data_chunk

    def __len__(self):
        return len(self.record_map)

    @staticmethod
    def unpack_single_tfrecord_file_from_path(tf_record_filename):
        """ Loads a tf-record file from the given path. Picks only every tenth frame to reduce the dataset and increase
        the diversity of it. With camera_segmentation_only = True, every availabe frame is picked (every tenth is annotated)
        Args:
            tf_record_filename: full path and name of the tf_record file to open
        Returns: a list of frames from the file
        """
        dataset = tf.data.TFRecordDataset(tf_record_filename, compression_type='')
        frame_list = {}
        frame_num = 0
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # if frame.images[0].camera_segmentation_label.panoptic_label:
            # if len(frame.camera_labels) != 0 and len(frame.laser_labels) != 0:
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                frame_list[frame_num] = frame
            frame_num += 1
        return frame_list
