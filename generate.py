import os
import yaml
from PIL import Image
import pandas as pd
from typing import List
from datetime import datetime
from libraries import remove_far_points, remove_ground, StixelGenerator, Stixel, remove_line_of_sight, remove_pts_below_plane_model

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
    for config_phase in ['testing']:
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
        except:
            continue
        if data_chunk is None:
            break
        for frame in data_chunk:
            try:
                lp_without_ground, plane_model = remove_ground(frame.points,
                                                               dataloader.config['rm_gnd'])
                lidar_pts = remove_far_points(lp_without_ground,
                                              dataloader.config['rm_far_pts'])
                lidar_pts = remove_pts_below_plane_model(lidar_pts,
                                                         plane_model)
                lidar_pts = remove_line_of_sight(lidar_pts, frame.camera_info.extrinsic.xyz,
                                                 dataloader.config['rm_los'])
                # camera is direct under lidar, no los
                stixel_gen = StixelGenerator(camera_info=frame.camera_info,
                                             img_size=dataloader.img_size,
                                             plane_model=plane_model,
                                             stixel_width=config['grid_step'],
                                             stixel_param=dataloader.config['stixel_cluster'],
                                             angle_param=dataloader.config['group_angle'])
                stixel_list = stixel_gen.generate_stixel(lidar_pts)
                # Export a single Stixel Wold representation and the relative images
                _export_single_dataset(image_left=frame.image,
                                       image_right=frame.image_right if dataloader.stereo_available else None,
                                       stixels=stixel_list,
                                       dataset_name=dataloader.name,
                                       name=f"{frame.name}_{str(frame_num)}",
                                       export_phase=phase)
                # print(f"Frame {frame_num + 1} from {len(data_chunk)} done.")
                frame_num += 1
            except Exception as e:
                print(f"{frame.name}_{str(frame_num)} failed due to {e}.")
                continue
        print(f"Record-file with idx {index + 1}/ {len(dataloader)} ({round(100/len(dataloader)*(index + 1), 1)}%) finished with {int(frame_num/1)} frames")
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))
    with open(f"failures_{phase}.txt", "a") as file:
        file.write(f"  . \n")


def _export_single_dataset(image_left: Image, stixels: List[Stixel], name: str, dataset_name: str, export_phase: str, image_right: Image = None):
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
    base_path = os.path.join(config['data_path'], dataset_name, export_phase)
    os.makedirs(base_path, exist_ok=True)
    left_img_path: str = os.path.join(base_path, "FRONT")
    right_img_path: str = os.path.join(base_path, "STEREO_RIGHT")
    label_path = os.path.join(base_path, "Stixel")
    os.makedirs(left_img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    # save images
    image_left.save(os.path.join(left_img_path, name + ".png"))
    if image_right is not None and export_phase == 'testing':
        os.makedirs(right_img_path, exist_ok=True)
        image_right.save(os.path.join(right_img_path, name + ".png"))
    # create .csv
    target_list = []
    for stixel in stixels:
        target_list.append([f"{export_phase}/{name}.png",
                            int(stixel.column),
                            int(stixel.top_row),
                            int(stixel.bottom_row),
                            round(stixel.depth, 1)])
    target: pd.DataFrame = pd.DataFrame(target_list)
    target.columns = ['img', 'u', 'vT', 'vB', 'd']
    # save .csv
    target.to_csv(os.path.join(label_path, name+".csv"), index=False)


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


if __name__ == "__main__":
    main()
