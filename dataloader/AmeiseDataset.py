import ameisedataset as ad
from ameisedataset.data import Camera, Lidar
from typing import List, Tuple
import glob
import os
import numpy as np
from PIL import Image

from libraries.names import point_dtype


class AmeiseData:
    def __init__(self, ad_frame: ad.data.Frame, ad_info: ad.data.Infos, name: str):
        """
        Base class for data from waymo open dataset
        Args:
            ad_frame:
            ad_info:
            name:
        """
        self.name: str = name
        self.frame: ad.data.Frame = ad_frame
        self.frame_info: ad.data.Infos = ad_info
        self.image: Image = ad.utils.transformation.rectify_image(image=self.frame.cameras[Camera.STEREO_LEFT],
                                                                  camera_information=self.frame_info.cameras[Camera.STEREO_LEFT],
                                                                  crop=True)
        self.image_right: Image = ad.utils.transformation.rectify_image(image=self.frame.cameras[Camera.STEREO_RIGHT],
                                                                        camera_information=self.frame_info.cameras[Camera.STEREO_RIGHT],
                                                                        crop=True)
        self.pov: np.array = ad_info.cameras[Camera.STEREO_LEFT].extrinsic.xyz
        self.points: np.array = self.point_slices()
        # transformation

    def point_slices(self) -> np.array:
        pts, projection = ad.utils.transformation.get_projection_matrix(pcloud=self.frame.lidar[Lidar.OS1_TOP].points,
                                                                        lidar_info=self.frame_info.lidar[Lidar.OS1_TOP],
                                                                        cam_info=self.frame_info.cameras[Camera.STEREO_LEFT])
        projection_list = np.array(projection)
        pts_coordinates = np.array(list(zip(pts['x'], pts['y'], pts['z'])))
        combined_data = np.hstack((pts_coordinates, projection_list))
        return np.array([tuple(row) for row in combined_data], dtype=point_dtype)   # x, y, z, proj_x, proj_y


class AmeiseDataLoader:
    def __init__(self, data_dir: str, phase: str, first_only: bool = False):
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
        self.name: str = "ameise-dataset"
        self.data_dir = os.path.join(data_dir, "ameise", phase)
        self.ameise_record_map: List[str] = glob.glob(os.path.join(self.data_dir, '*.4mse'))
        self.first_only: bool = first_only
        self.img_size = {'width': 1920, 'height': 1200}
        self.stereo_available: bool = True
        print(f"Found {len(self.ameise_record_map)} Ameise record files.")

    def __getitem__(self, idx: int) -> List[AmeiseData]:
        try:
            infos, frames = ad.unpack_record(self.ameise_record_map[idx])
            print(infos.filename)
        except:
            print(self.ameise_record_map[idx])
            return None
        ameise_data_chunk: List[AmeiseData] = []
        frame_num: int = 0
        for ad_frame in frames:
            if frame_num % 2 == 0:      # just use every second frame: 5 Hz
                if ad_frame.cameras[1].image is not None and ad_frame.cameras[2].image is not None:
                    ameise_data_chunk.append(AmeiseData(ad_frame, infos, name=f"frame_{str(idx)}_{self.ameise_record_map[idx].split('/')[-1].split('.')[0]}-{frame_num}" ))
                    if self.first_only:
                        break
            frame_num += 1
        return ameise_data_chunk

    def __len__(self) -> int:
        return len(self.ameise_record_map)
