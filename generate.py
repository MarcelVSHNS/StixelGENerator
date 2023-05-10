import os
import math
import yaml
import threading
import pandas as pd
from zipfile import ZipFile
import os
from pathlib import Path
from os.path import basename

from dataloader.WaymoDataset import WaymoDataLoader
from libraries.pointcloudlib import get_stixel_from_laser_data
from libraries.pointcloudlib import force_stixel_into_image_grid


# open Config
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
data_dir = config['raw_data_path'] + config['phase']


def main():
    # load data - provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
    #     .images (5 views) and .laser_points (top lidar, divided into 5 fitting views).
    waymo_dataset = WaymoDataLoader(data_dir=data_dir, camera_segmentation_only=False, first_only=False)
    # Usage: sample = waymo_dataset[idx][frame_num]

    thread_workload = int(len(waymo_dataset) / config['num_threads'])
    assert thread_workload > 0, "Too less files for num_threads"
    # distribution of idx over the threads
    dataset_chunk = list(chunks(list(range(len(waymo_dataset))), thread_workload))
    threads = []
    for file_index_list in dataset_chunk:
        # create thread with arg dataset_list (chunk)
        x = threading.Thread(target=thread__generate_data_from_tfrecord_chunk, args=(file_index_list, waymo_dataset))
        threads.append(x)
        x.start()
        print("Thread is working ...")
    for thread in threads:
        thread.join()
    create_sample_map()
    create_zip_chunks()
    print("Finished!")


def thread__generate_data_from_tfrecord_chunk(index_list, dataloader):
    # work on a list of assigned indices
    for index in index_list:
        # iterate over all frames inside the tfrecord
        frame_num = 0
        for frame in dataloader[index]:
            # if frame_num % 5 == 0:
            # iterate over all needed views
            laser_stixel, laser_by_angle = get_stixel_from_laser_data(
                laser_points_by_view=frame.laser_points[:config['num_views']])
            training_data = force_stixel_into_image_grid(laser_stixel)
            for camera_view in range(len(training_data)):
                export_single_dataset(image=frame.images[camera_view],
                                      stixels=training_data[camera_view],
                                      name=f"{frame.name}-{frame_num}-{camera_view}")
            frame_num += 5
        print(f"TFRecord-file with idx {index} finished with {frame_num/5} frames")


def export_single_dataset(image, stixels, name):
    img_path = os.path.join(config['data_path'], config['phase'])
    label_path = os.path.join(img_path, config['targets'])
    # save image
    image.save(os.path.join(img_path, name+".png"))
    # create gt line
    target_list = []
    for stixel in stixels:
        depth = math.sqrt(math.pow(stixel[0], 2) + math.pow(stixel[1], 2))
        target_list.append([f"{config['phase']}/{name}.png", int(stixel[3]), int(stixel[4]), int(0.0), round(depth, 1)])
    target = pd.DataFrame(target_list)
    target.columns = ['img_path', 'x', 'y', 'class', 'depth']
    target.to_csv(os.path.join(label_path, name+".csv"), index=False)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def create_sample_map():
    files = os.listdir(f"data/{config['phase']}/targets")
    map = []
    for file in files:
        map.append(os.path.splitext(file)[0])
    image_map = pd.DataFrame(map)
    image_map.to_csv(f"data/{config['phase']}.csv", index=False, header=False)
    print("Map created")


def create_zip_chunks():
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
    image_chunks = chunks(image_list, n_sized)
    annotZip = ZipFile(os.path.join(output_path, phase + '_' + data_set + '_compressed_annotations_' + '.zip'), 'w')

    n = 0
    for img_list in image_chunks:
        # create a ZipFile object
        with ZipFile(os.path.join(output_path, phase + '_' + data_set + '_compressed_' + str(n) + '.zip'), 'w') as zipObj:
            for img in img_list:
                # create complete filepath of file in directory
                file_path = os.path.join(input_image_path, img)
                zipObj.write(file_path, basename(file_path))
                if single_stixel_pos_export:
                    annot = os.path.splitext(img)[0] + '.csv'
                    annot_file_path = os.path.join(input_annotations_path, annot)
                    annotZip.write(annot_file_path, basename(annot_file_path))
        n += 1
        print("zip num: " + str(n) + " from total " + str(num_packages) + " created!")
    annotZip.close()


if __name__ == "__main__":
    main()
