import numpy as np
import open3d as o3d
import yaml

from dataloader.WaymoDataset import WaymoDataLoader
from libraries.visualization import plot_points_on_image
from libraries.pointcloudlib import get_stixel_from_laser_data
from libraries.pointcloudlib import detect_objects_in_point_cloud_numerical
from libraries.visualization import plot_points_2d_graph
from libraries.pointcloudlib import analyse_lidar_col_for_stixel
from libraries.pointcloudlib import force_stixel_into_image_grid


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
    idx = config['exploring']['idx']
    frame_num = config['exploring']['frame_num']
    view = config['exploring']['view']
    col = config['exploring']['col']

    assert len(waymo_dataset) > idx, f'The index is too high, records found: {len(waymo_dataset)}'
    assert len(waymo_dataset[idx][frame_num].images) != 0, 'The chosen index has no segmentation data'
    sample = waymo_dataset[idx][frame_num]

    # Show the Objects by point cloud
    # detect_objects_in_point_cloud_numerical(sample.laser_points[view][..., :3], visualize=True)

    # Show the view with points
    if config['explore_data_deep']:
        # usage of the visualization lib
        for i in range(1):
            plot_points_on_image(images=sample.images[i],
                                 laser_points=sample.laser_points[i])

    # Basic concept is to load general data like waymo and apply the library functions to the provided data
    # Get Stixel from Laser
    # Get Laser points sorted by angle
    laser_stixel, laser_by_angle = get_stixel_from_laser_data(laser_points_by_view=sample.laser_points[:config['num_views']])

    # Draw Stixel on image
    if config['explore_data']:
        #stixels, _reason = analyse_lidar_col_for_stixel(laser_by_angle[view][col], investigate=True)
        plot_points_on_image(images=sample.images[view],
                             laser_points=np.vstack(laser_stixel[view]))
        #plot_points_on_image(images=sample.images[view],
        #                     laser_points=laser_by_angle[view][col],
        #                     stixels=np.hstack((stixels, _reason)))
        #plot_points_2d_graph(laser_points_by_angle=laser_by_angle[view][col],
        #                     stixels=np.hstack((stixels, _reason)),
        #                     title=str(col))

        training_data = force_stixel_into_image_grid(laser_stixel)
        plot_points_on_image(images=sample.images[view],
                             laser_points=training_data[view])


if __name__ == "__main__":
    main()
