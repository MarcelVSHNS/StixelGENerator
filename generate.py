import os
import math
import yaml
from PIL import Image
from threading import Thread
import pandas as pd
import numpy as np
from typing import List, Tuple
from zipfile import ZipFile
from pathlib import Path
from os.path import basename
from datetime import datetime
# change xxxDataLoader to select the dataset
from dataloader import AmeiseDataLoader as Dataset, AmeiseData as Data
from libraries import (remove_far_points, remove_ground, group_points_by_angle, Scanline, force_stixel_into_image_grid,
                       PositionClass, Stixel, remove_line_of_sight)

# open Config
with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
overall_start_time = datetime.now()
phase = ''


def main():
    # config['phases']
    for config_phase in ['testing']:
        global phase
        phase = config_phase
        # load data - provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
        #     .images (5 views) and .laser_points (top lidar, divided into 5 fitting views).
        dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=phase, first_only=False)
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
        # create_zip_chunks()
        overall_time = datetime.now() - overall_start_time
        print(f"Finished {phase} set! in {str(overall_time).split('.')[0]}")


def _generate_data_from_record_chunk(index_list: List[int], dataloader: Dataset):
    # work on a list of assigned indices
    for index in index_list:
        print(f'index: {index} in progress ...')
        # iterate over all frames inside the tfrecord
        frame_num: int = 0
        data_chunk: List[Data] = dataloader[index]
        if data_chunk is None:
            break
        for frame in data_chunk:
            lp_without_ground = remove_ground(frame.points)
            lp_without_far_pts: np.array = remove_far_points(lp_without_ground)
            lp_without_los = remove_line_of_sight(lp_without_far_pts, frame.camera_pov)
            lp_wg_ordered_by_angle = group_points_by_angle(lp_without_los)
            stixel: List[List[Stixel]] = []
            for laser_points_by_angle in lp_wg_ordered_by_angle:
                column: Scanline = Scanline(laser_points_by_angle)
                stixel.append(column.get_stixels())                     # calculate stixel
            grid_stixel: List[Stixel] = force_stixel_into_image_grid(stixel, dataloader.img_size)
            _export_single_dataset(image_left=frame.image,
                                   image_right=frame.image_right if dataloader.stereo_available else None,
                                   stixels=grid_stixel,
                                   dataset_name=dataloader.name,
                                   name=frame.name)
            frame_num += 1
        print(f"Record-file with idx {index + 1}/ {len(index_list)} ({round(100/len(index_list)*(index + 1), 1)}%) finished with {int(frame_num/1)} frames")
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))


def _export_single_dataset(image_left: Image, stixels: List[Stixel], name: str, dataset_name: str, image_right: Image = None):
    # define paths
    base_path = os.path.join(config['data_path'], dataset_name, phase)
    os.makedirs(base_path, exist_ok=True)
    left_img_path: str = os.path.join(base_path, "STEREO_LEFT")
    right_img_path: str = os.path.join(base_path, "STEREO_RIGHT")
    label_path = os.path.join(base_path, "targets_from_lidar")
    os.makedirs(left_img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    # save images
    image_left.save(os.path.join(left_img_path, name + ".png"))
    if image_right is not None:
        os.makedirs(right_img_path, exist_ok=True)
        image_right.save(os.path.join(right_img_path, name + ".png"))
    # create .csv
    test = PositionClass.BOTTOM
    target_list = []
    for stixel in stixels:
        stixel_class: int = stixel.position_class.value
        depth: float = math.sqrt(math.pow(stixel.point['x'], 2) + math.pow(stixel.point['y'], 2))
        target_list.append([f"{phase}/{name}.png", int(stixel.point['proj_x']), int(stixel.point['proj_y']), int(stixel_class), round(depth, 1)])
    target: pd.DataFrame = pd.DataFrame(target_list)
    target.columns = ['img_path', 'x', 'y', 'class', 'depth']
    # save .csv
    target.to_csv(os.path.join(label_path, name+".csv"), index=False)


def _chunks(lst, n) -> List[List[int]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _create_zip_chunks():
    # TODO: not ready yet!!
    output_path = os.path.join(os.getcwd(), config['data_path'], config['phase'] + "_compressed")
    input_image_path = os.path.join(os.getcwd(), config['data_path'], config['phase'])
    input_annotations_path = os.path.join(input_image_path, "targets")
    data_set = config['dataset_name']
    num_packages = config[f"num_zips_{config['phase']}"]
    phase = config['phase']

    single_stixel_pos_export = os.path.isdir(input_annotations_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    img_folder_list = os.listdir(input_image_path)
    image_list = [f for f in img_folder_list if f.endswith('.png')]
    n_sized = int(len(image_list) / num_packages)
    image_chunks = _chunks(image_list, n_sized)
    annotZip = ZipFile(os.path.join(output_path, phase + '_' + data_set + '_compressed_annotations_' + '.zip'), 'w')

    n = 0
    for img_list in image_chunks:
        # create a ZipFile object
        with ZipFile(os.path.join(output_path, phase + '_' + data_set + '_compressed_' + str(n) + '.zip'), 'w') as zipObj:
            for img in img_list:
                # create complete filepath of file in directory
                file_path = os.path.join(input_image_path, str(img))
                zipObj.write(file_path, basename(file_path))
                if single_stixel_pos_export:
                    annot = os.path.splitext(str(img))[0] + '.csv'
                    annot_file_path = os.path.join(input_annotations_path, annot)
                    annotZip.write(annot_file_path, basename(annot_file_path))
        n += 1
        print("zip num: " + str(n) + " from total " + str(num_packages) + " created!")
    annotZip.close()


if __name__ == "__main__":
    main()
