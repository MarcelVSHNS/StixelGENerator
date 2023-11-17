import numpy as np
import pykitti
import os
from typing import List, Tuple

class KittiData:
    def __init__(self, frame, name):
        self.name: str = name
        self.frame = frame

        self.image = None
        self.image_right = None
        self.pov = None
        self.points = None


class KittiDataLoader:
    def __init__(self, data_dir, phase, first_only=False):
        self.name: str = "KITTI-dataset"
        self.data_dir = os.path.join(data_dir, "kitti")
        self.kitti_record_map: np.array = self._read_kitty_data_structure()
        self.first_only: bool = first_only
        self.img_size = {'width': 1392, 'height': 512}
        self.stereo_available: bool = False
        print(f"Found {len(self.kitti_record_map)} Kitti record files.")

    def __getitem__(self, idx: int) -> List[KittiData]:
        frames = pykitti.raw(base_path=self.data_dir,
                             date=self.kitti_record_map[idx]['date'],
                             drive=self.kitti_record_map[idx]['drive'])

        kitti_data_chunk: List[KittiData] = []
        frame_num: int = 0
        for frame in frames:
            name = f"set_{str(idx)}_{self.kitti_record_map[idx]['drive']}_{frame_num}"
            kitti_data_chunk.append(KittiData(frame, name=name))
            if self.first_only:
                break
            frame_num += 1
        return kitti_data_chunk

    def __len__(self) -> int:
        return len(self.kitti_record_map)

    def _read_kitty_data_structure(self) -> np.array:
        date_drive_list = []
        for date_folder in os.listdir(self.data_dir):
            date_folder_path = os.path.join(self.data_dir, date_folder)
            if os.path.isdir(date_folder_path):
                for drive_folder in os.listdir(date_folder_path):
                    drive_folder_path = os.path.join(date_folder_path, drive_folder)
                    if os.path.isdir(drive_folder_path):
                        date_drive_list.append([date_folder, drive_folder])
        d_type = [('date', 'U15'), ('drive', 'U30')]
        return np.array(date_drive_list, dtype=d_type)
