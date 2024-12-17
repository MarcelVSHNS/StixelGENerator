"""
TODO: check U column range (found -1 as u)
"""

import os
import numpy as np
import yaml
import stixel as stx
from PIL import Image
import cv2
import pandas as pd
from typing import List
from datetime import datetime

from stixel.definition import StixelWorld

from libraries import remove_far_points, remove_ground, StixelGenerator, Stixel, remove_line_of_sight, \
    remove_pts_below_plane_model, filter_points_by_label, filter_points_by_semantic, group_points_by_angle, \
    calculate_plane_from_bbox, segment_ground

# open Config
with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
if config['dataset'] == "waymo":
    from dataloader import WaymoDataLoader as Dataset, WaymoData as Data
elif config['dataset'] == "kitti":
    from dataloader import KittiDataLoader as Dataset, KittiData as Data
else:
    raise ValueError("Dataset not supported")
overall_start_time = datetime.now()


def main():
    """
    Basic start of the project. Configure phase and load the related data. E.g. Kitti dataloader provides datasets
    organised by drive.
    """
    # config['phases']      'validation', 'testing', 'training'
    for config_phase in ['training', 'validation']:
        phase = config_phase
        with open(f"failures_{phase}.txt", "w") as file:
            file.write("Record names by phase, which failed to open: \n")
        # every dataset consist all frames of one drive (kitti)
        dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=phase, first_only=False)
        """ Multi Threading - deprecated
        process_workload: int = int(len(dataset) / config['num_threads'])
        assert process_workload > 0, "Too less files for num_threads"
        # distribution of idx over the threads
        dataset_chunk: List[List[int]] = list(_chunks(list(range(len(dataset))), process_workload))
        processes: List[Thread] = []
        for file_index_list in dataset_chunk:
            # create thread with arg dataset_list (chunk)
            x: Thread = Thread(target=_generate_data_from_record_chunk, args=(file_index_list, dataset))
            processes.append(x)
            x.start()
            print("Process is working ...")
        for process in processes:
            process.join()
        """
        _generate_data_from_record_chunk(list(range(len(dataset))), dataset, phase=phase)
        overall_time = datetime.now() - overall_start_time
        print(f"Finished {phase} set! in {str(overall_time).split('.')[0]}")


def _generate_data_from_record_chunk(index_list: List[int], dataloader: Dataset, phase: str):
    """
    Iterates through all drives, frame by frame. For every frame a stixel world is generated.
    The access is simply done by an index list which enables the distribution to multiple threads
    Args:
        index_list: A list of indices representing the records to be processed.
        dataloader: The dataset object that provides access to the data.
        phase: The phase of data processing.
    """
    with open(f"failures_{phase}.txt", "a") as file:
        file.write(f"gsutil -m cp \\ \n")
    # work on a list of assigned indices
    for index in index_list:
        print(f'index: {index} in progress ...')
        # iterate over all frames inside the record
        frame_num: int = 0
        try:
            data_chunk: List[Data] = dataloader[index]
        except Exception as e:
            print(e)
            continue
        if data_chunk is None:
            break
        for frame in data_chunk:
            try:
                stixel_list = []
                stixel_gen = StixelGenerator(camera_info=frame.camera_info,
                                             img_size=dataloader.img_size,
                                             stixel_width=config['grid_step'],
                                             stixel_param=dataloader.config['stixel_cluster'],
                                             angle_param=dataloader.config['group_angle'])
                if config['generator'] == 'default':
                    lidar_pts = default_generator(frame, dataloader)
                    stixel_list = stixel_gen.generate_stixel(lidar_pts)
                elif config['generator'] == 'bbox':
                    lidar_pts, bbox_ids = bbox_generator(frame, dataloader)
                    stixel_list = []
                    for idx in bbox_ids:
                        for bbox in frame.laser_labels:
                            if bbox.id == idx:
                                plane = calculate_plane_from_bbox(bbox)
                                stixel_gen.plane_model = plane
                                bbox_points = lidar_pts[lidar_pts['id'] == idx]
                                stixel_list.append(stixel_gen.generate_stixel(bbox_points))
                    stixel_list = [item for sublist in stixel_list for item in sublist]
                elif config['generator'] == 'semantic':
                    lidar_pts, plane_model = semantic_filtering_generator(frame, dataloader)
                else:
                    raise ValueError("Not recognized generator.")

                #

                # Export a single Stixel Wold representation and the relative images
                _export_single_dataset(stixels=stixel_list,
                                       frame=frame,
                                       dataset_name=dataloader.name,
                                       export_phase=phase)
                # print(f"Frame {frame_num + 1} from {len(data_chunk)} done.")
                frame_num += 1
            except Exception as e:
                print(f"{frame.name} failed due to {e}.")
                continue
        print(
            f"Record-file with idx {index + 1}/ {len(dataloader)} ({round(100 / len(dataloader) * (index + 1), 1)}%) finished with {int(frame_num / 1)} frames")
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))
    with open(f"failures_{phase}.txt", "a") as file:
        file.write(f"  . \n")


def _export_single_dataset(stixels: List[Stixel], frame: Data, dataset_name: str, export_phase: str):
    """
    Exports the Stixel World and cares for paths etc.
    Args:
        image_left: The left stereo image.
        stixels: A list of stixels representing objects in the image.
        name: The name of the image dataset.
        dataset_name: The name of the dataset.
        export_phase: The phase of the dataset to export (e.g., training, testing).
        image_right: The right stereo image (optional, only required for testing phase).
    """
    # define paths
    base_path = os.path.join(config['data_path'], dataset_name, f"Stixel_{config['generator']}")
    os.makedirs(base_path, exist_ok=True)
    # left_img_path: str = os.path.join(base_path, "FRONT")
    label_path = os.path.join(base_path, export_phase)
    # os.makedirs(left_img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    stxl_wrld = stx.StixelWorld()
    stxl_wrld.context.name = frame.name
    stxl_wrld.context.calibration.K.extend(frame.camera_info.K.flatten().tolist())
    stxl_wrld.context.calibration.T.extend(frame.camera_info.T.flatten().tolist())
    stxl_wrld.context.calibration.reference = "Vehicle2Camera"
    stxl_wrld.context.calibration.R.extend(frame.camera_info.R.flatten().tolist())
    stxl_wrld.context.calibration.D.extend(frame.camera_info.D.flatten().tolist())
    stxl_wrld.context.calibration.DistortionModel = 1
    stxl_wrld.context.calibration.img_name = f"{frame.name}.png"
    height, width, channels = np.array(frame.image).shape
    stxl_wrld.context.calibration.width = width
    stxl_wrld.context.calibration.height = height
    stxl_wrld.context.calibration.channels = channels
    # save images
    img = np.array(frame.image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, img_encoded = cv2.imencode(".png", img)
    stxl_wrld.image = img_encoded.tobytes()
    # if image_right is not None and export_phase == 'testing':
    #     os.makedirs(right_img_path, exist_ok=True)
    #     image_right.save(os.path.join(right_img_path, name + ".png"))
    # create .stx1
    for stixel in stixels:
        stxl = stx.Stixel()
        stxl.u = int(stixel.column)
        stxl.vT = int(stixel.top_row)
        stxl.vB = int(stixel.bottom_row)
        stxl.d = round(stixel.depth, 3)
        stxl.label = int(stixel.sem_seg)
        stxl.width = 8
        stxl.confidence = 1.0
        stxl_wrld.stixel.append(stxl)
    stx.save(stxl_wrld, label_path)


def _chunks(lst, n) -> List[List[int]]:
    """
    Args:
        lst: A list of integers or other data types.
        n: An integer representing the size of each chunk.
    Returns:
        A list of lists, where each sublist contains 'n' elements from the input list 'lst'. The last sublist may
        contain fewer than 'n' elements if the length of the input list is not divisible by 'n'.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def default_generator(frame, dataloader, remove_los: bool = False) -> np.array:
    angled_pts = group_points_by_angle(points=frame.points, param=dataloader.config['group_angle'],
                                       camera_info=frame.camera_info)
    lp_without_ground = segment_ground(angled_pts)
    lidar_pts = remove_far_points(lp_without_ground,
                                  dataloader.config['rm_far_pts'])
    if remove_los:
        lidar_pts = remove_line_of_sight(lidar_pts, frame.camera_info.extrinsic.xyz,
                                         dataloader.config['rm_los'])
    return lidar_pts


def bbox_generator(frame, dataloader):
    angled_pts = group_points_by_angle(points=frame.points, param=dataloader.config['group_angle'],
                                       camera_info=frame.camera_info)
    lp_without_ground, _ = remove_ground(points=angled_pts,
                                         param=dataloader.config['rm_gnd'])
    pts_filter_bbox, bbox_ids = filter_points_by_label(points=lp_without_ground,
                                                       bboxes=frame.laser_labels)
    pts_filter_bbox = remove_far_points(points=pts_filter_bbox,
                                        param=dataloader.config['rm_far_pts'])
    return pts_filter_bbox, bbox_ids


def semantic_filtering_generator(frame, dataloader):
    pts_filter_sem_seg = filter_points_by_semantic(points=frame.points,
                                                   param=dataloader.config['semantic_filter'])
    lp_without_ground, ground_model = remove_ground(points=frame.points,
                                                    param=dataloader.config['rm_gnd'])
    pts_filter_bbox = remove_far_points(points=pts_filter_sem_seg,
                                        param=dataloader.config['rm_far_pts'])
    return pts_filter_bbox, ground_model


if __name__ == "__main__":
    main()
