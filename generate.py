import os
import yaml
from PIL import Image
import pandas as pd
from typing import List, Tuple
from datetime import datetime
from dataloader import KittiDataLoader as Dataset, KittiData as Data
from libraries import remove_far_points, remove_ground, StixelGenerator, Stixel, remove_line_of_sight, remove_pts_below_plane_model

# open Config
with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
overall_start_time = datetime.now()


def main():
    """
    Basic start of the project. Configure phase and load the related data. E.g. Kitti dataloader provides datasets
    organised by drive.
    """
    # config['phases']      , 'validation', 'testing'
    for config_phase in ['training', 'validation', 'testing']:
        phase = config_phase
        # every dataset consist all frames of one drive (kitti)
        dataset: Dataset = Dataset(data_dir=config['raw_data_path'], phase=phase, first_only=True)
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
    # work on a list of assigned indices
    for index in index_list:
        print(f'index: {index} in progress ...')
        # iterate over all frames inside the record
        frame_num: int = 0
        data_chunk: List[Data] = dataloader[index]

        if data_chunk is None:
            break
        for frame in data_chunk:
            lp_without_ground, plane_model = remove_ground(frame.points)
            lp_without_far_pts = remove_far_points(lp_without_ground)
            lp_plane_model_corrected = remove_pts_below_plane_model(lp_without_far_pts, plane_model)
            lp_without_los = remove_line_of_sight(lp_plane_model_corrected, frame.camera_pov)
            stixel_gen = StixelGenerator(camera_info=frame.camera_info, img_size=dataloader.img_size,
                                         plane_model=plane_model)
            stixel_list = stixel_gen.generate_stixel(lp_without_los)
            # Export a single Stixel Wold representation and the relative images
            _export_single_dataset(image_left=frame.image,
                                   image_right=frame.image_right if dataloader.stereo_available else None,
                                   stixels=stixel_list,
                                   dataset_name=dataloader.name,
                                   name=frame.name,
                                   export_phase=phase)
            frame_num += 1
        print(f"Record-file with idx {index + 1}/ {len(index_list)} ({round(100/len(index_list)*(index + 1), 1)}%) finished with {int(frame_num/1)} frames")
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))


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
    left_img_path: str = os.path.join(base_path, "STEREO_LEFT")
    right_img_path: str = os.path.join(base_path, "STEREO_RIGHT")
    label_path = os.path.join(base_path, "targets_from_semseg")
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
                            int(stixel.position_class.value),
                            round(stixel.depth, 1)])
    target: pd.DataFrame = pd.DataFrame(target_list)
    target.columns = ['img_path', 'x', 'yT', 'yB', 'class', 'depth']
    # save .csv
    target.to_csv(os.path.join(label_path, name+".csv"), index=False)


def _chunks(lst, n) -> List[List[int]]:
    """
    Args:
        lst: A list of integers or other data types.
        n: An integer representing the size of each chunk.
    Returns:
        A list of lists, where each sublist contains 'n' elements from the input list 'lst'. The last sublist may contain fewer than 'n' elements if the length of the input list is not divisible by 'n'.
    Example:
        >>> _chunks([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    main()
