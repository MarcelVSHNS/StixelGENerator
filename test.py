import numpy as np
import yaml

from dataloader.WaymoDataset import WaymoDataLoader
from libraries.visualization import plot_points_on_image
from libraries.pointcloudlib import get_stixel_from_laser_data


def main():
    # open Config
    with open('config.yaml') as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    data_dir = config['raw_data_path'] + config['phase']

    """ load data - provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
    .images (5 views) and .laser_points (top lidar, divided into 5 fitting views). Like e.g.:
        798 tfrecord-files (selected by "idx")
            ~20 Frames (batch size/ dataset - selected by "frame_num")
                5 .images (camera view - selected by index[])
                5 .laser_points (shape of [..., [x, y, z, img_x, img_y]])"""
    waymo_dataset = WaymoDataLoader(data_dir=data_dir, camera_segmentation_only=False, first_only=True)

    # Example to display the lidar camera projection and provide an exploring data sample
    idx = 0
    frame_num = -1
    assert len(waymo_dataset) > idx, f'The index is too high, records found: {len(waymo_dataset)}'
    assert len(waymo_dataset[idx][frame_num].images) != 0, 'The chosen index has no segmentation data'
    sample = waymo_dataset[idx][frame_num]

    """
    if config['explore_data']:
        # usage of the visualization lib
        for i in range(5):
            plot_points_on_image(images=sample.images[i],
                                 laser_points=sample.laser_points[i])
    """

    # Basic concept is to load general data like waymo and apply the library functions to the provided data
    # Get Stixel from Laser
    laser_stixel = get_stixel_from_laser_data(laser_points_by_view=sample.laser_points)
    # Draw Stixel on image
    if config['explore_data']:
        view = 0
        plot_points_on_image(images=sample.images[view],
                             laser_points=laser_stixel[view][399])



if __name__ == "__main__":
    main()
