import os
from typing import List, Tuple
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from libraries.names import point_dtype


class DelftData:
    def __init__(self, frame, name):
        self.name: str = name
        self.frame = frame

        self.image = None
        self.image_right = None
        self.pov = None
        self.points = None


class DelftDataLoader:
    def __init__(self, data_dir, phase, first_only=False):
        self.name: str = "ViewOfDelft-dataset"
        self.data_dir = os.path.join(data_dir, "delft", phase)
        self.kitti_locations = KittiLocations(root_dir=data_dir + "delft")
        delf_record_map_path = os.path.join(data_dir,"lidar","ImageSets", phase + ".txt")
        self.delft_record_map: List[str] = [line.strip() for line in open(delf_record_map_path)]
        self.first_only: bool = first_only
        self.img_size = {'width': 1936, 'height': 1216}
        self.stereo_available: bool = False
        print(f"Found {len(self.delft_record_map)} VoD record files.")

    def __getitem__(self, idx: int) -> List[DelftData]:
        vod_frame = FrameDataLoader(kitti_locations=self.kitti_locations,
                                frame_number=self.delft_record_map[idx])
        vod_data = DelftData(vod_frame, name=f"frame_{self.delft_record_map[idx]}" )
        return [vod_data]

    def __len__(self) -> int:
        return len(self.delft_record_map)
