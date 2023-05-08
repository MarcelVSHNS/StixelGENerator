import os
import math
import yaml
import threading
import pandas as pd

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
    waymo_dataset = WaymoDataLoader(data_dir=data_dir, camera_segmentation_only=False, first_only=True)
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
    print("Finished!")


def thread__generate_data_from_tfrecord_chunk(index_list, dataloader):
    # work on a list of assigned indices
    for index in index_list:
        # iterate over all frames inside the tfrecord
        frame_num = 0
        for frame in dataloader[index]:
            if frame_num % 5 == 0:
                # iterate over all needed views
                laser_stixel, laser_by_angle = get_stixel_from_laser_data(
                    laser_points_by_view=frame.laser_points[:config['num_views']])
                training_data = force_stixel_into_image_grid(laser_stixel)
                for camera_view in range(len(training_data)):
                    export_single_dataset(image=frame.images[camera_view],
                                          stixels=training_data[camera_view],
                                          name=f"{frame.name}-{frame_num}-{camera_view}")
            frame_num += 1


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


if __name__ == "__main__":
    main()
