import ameisedataset as ad
from ameisedataset.data import Camera, Lidar

import glob
import os
import numpy as np
import time

from typing import List

class AmeiseDataLoader:
    def __init__(self, data_dir, first_only=False):
        """
        Loads a full set of ameise data in single frames, can be one .4mse file or a folder of .4mse files.
        provides a list by index for a .4mse-file which has ~50 frame objects. Every object has lists of
        .images (4 views) and .lidar (top lidar, divided into 5 fitting views). Like e.g.:
        798 .4mse-files (selected by "idx")
            ~50 Frames (batch size/ dataset - selected by "frame_num")
                4 .images (camera view - selected by index[])
                3 .laser_points (shape of [..., [x, y, z, img_x, img_y]])
        Args:
            data_dir: specify the location of the tf_records
            first_only: doesn't load the full ~20 frames to return a data sample if True
        """
        self.data_dir = data_dir
        self.ameise_record_map = glob.glob(os.path.join(self.data_dir, '*.4mse'))
        self.first_only = first_only
        print(f"Found {len(self.ameise_record_map)} record files")

    def __getitem__(self, idx):
        try:
            infos, frames = ad.unpack_record(self.ameise_record_map[idx])
            print(infos.filename)
        except:
            print(self.ameise_record_map[idx])
            return None
        ameise_data_chunk = []
        frame_num = 0
        for ad_frame in frames:
            # start_time = datetime.now()
            if frame_num % 2 == 0:
                if ad_frame.cameras[1].image is not None and ad_frame.cameras[2].image is not None:
                    ameise_data_chunk.append(AmeiseData(ad_frame, infos, name="frame_" + str(idx) + "_" + self.ameise_record_map[idx].split('/')[-1].split('.')[0]))
                    # self.object_creation_time = datetime.now() - start_time
                    if self.first_only:
                        break
            frame_num += 1
        return ameise_data_chunk

    def __len__(self):
        return len(self.ameise_record_map)

class AmeiseData(ad.data.Frame):
    def __init__(self, ad_frame: ad.data.Frame, ad_info, name):
        """
        Base class for data from waymo open dataset
        Args:
            ad_frame:
            ad_info:
            name:
        """
        super().__init__(ad_frame.frame_id, timestamp=ad_frame.timestamp)
        self.cameras: List[ad.data.Image] = ad_frame.cameras
        self.lidar: List[ad.data.Points] = ad_frame.lidar
        self.image_points = [[] for _ in range(4)]
        self.name = name
        # transformation
        self.point_slices(ad_info)
        self.rectify_images(ad_info)

    def rectify_images(self, info: ad.data.Infos):
        cams_available, _ = self.get_data_lists()
        for camera in cams_available:
            self.cameras[camera] = ad.utils.transformation.rectify_image(self.cameras[camera], info.cameras[camera], crop=True)

    def point_slices(self, info: ad.data.Infos):
        cams_available, _ = self.get_data_lists()
        if ad.data.Camera.STEREO_RIGHT in cams_available:
            cams_available.remove(ad.data.Camera.STEREO_RIGHT)
        for camera in cams_available:
            pts, projection = ad.utils.transformation.get_projection_matrix(pcloud=self.lidar[ad.data.Lidar.OS1_TOP].points,
                                                                            lidar_info=info.lidar[ad.data.Lidar.OS1_TOP],
                                                                            cam_info=info.cameras[camera])
            projection_list = np.array(projection)
            pts_coordinates = np.array(list(zip(pts['x'], pts['y'], pts['z'])))
            combined_data = np.hstack((pts_coordinates, projection_list))
            self.image_points[camera] = combined_data   # x, y, z, proj_x, proj_y
