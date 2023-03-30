import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from waymo_open_dataset import dataset_pb2 as open_dataset


class WaymoDataLoader(object):
    def __init__(self, data_dir, tf_record_files=None, camera_segmentation_only=False, shuffle=False):
        """
        Loads a full set of waymo data in single frames, can be one tf_record file or a folder of  tf_record files.
        Args:
            data_dir: specify the location of the tf_records
            tf_record_files: pick a specific file from data_dir
            camera_segmentation_only: if True, loads only frames with available camera segmentation
            shuffle: if True, the frames will be shuffled
        """
        self.data_dir = data_dir
        self.tf_record_files = os.path.join(data_dir, tf_record_files)
        self.camera_segmentation_only = camera_segmentation_only
        self.frames = []
        if self.tf_record_files:
            assert self.tf_record_files.endswith('.tfrecord')
            self.unpack_single_tf_record_file_from_path(self.tf_record_files)
        else:
            self.load_data_from_folder()
        print(f"Num_frames: {len(self.frames)}")

    def load_data_from_folder(self):
        data_folder = os.path.join(self.data_dir, '*.tfrecord')
        dataset_list = glob.glob(data_folder)
        print(f"Found {len(dataset_list)} record files")
        for dataset_name in dataset_list:
            # unpack every used frame from set and store in frame list
            self.unpack_single_tf_record_file_from_path(os.path.join(self.data_dir, dataset_name))

    def unpack_single_tf_record_file_from_path(self, tf_record_filename):
        """ Loads a tf-record file from the given path and returns a list of frames from the file """
        dataset = tf.data.TFRecordDataset(tf_record_filename, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if self.camera_segmentation_only:
                if frame.images[0].camera_segmentation_label.panoptic_label:
                    self.frames.append(frame)
            else:
                self.frames.append(frame)
            # if self.laser_segmentation_only:
                # if frame.lasers[0].ri_return1.segmentation_label_compressed:
                #    frame_list.append(frame)